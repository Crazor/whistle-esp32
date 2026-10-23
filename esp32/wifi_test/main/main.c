/* wifi_test - bring-up check for the Waveshare ESP32-P4-WIFI6's Wi-Fi.
 *
 * The P4 has no radio. Wi-Fi runs on the board's ESP32-C6, which the P4
 * reaches over SDIO (CLK=GPIO18 CMD=GPIO19 D0-D3=GPIO14-17, C6 reset on
 * GPIO54). esp_hosted carries the link and esp_wifi_remote gives the P4 the
 * usual esp_wifi_* API, so the code below is a plain IDF station.
 *
 * Each step checks one more piece of the chain:
 *   cp            the C6 answered over SDIO and reported its firmware
 *   scan          its radio works: the access points it hears
 *   got_ip        it joined the network and DHCP ran through the link
 *   dns, http     data flows both ways: a name lookup and an HTTP request
 *
 * The network is stored in NVS by `join` and joined again at every boot.
 *
 * In, one command per line on the console:
 *   scan                         list the access points in range
 *   join ssid=<s> [pass=<p>]     store a network and join it; quote values
 *                                with spaces: ssid="My Net"
 *   forget                       leave and erase the stored network
 *   status                       one STATUS line
 *   net                          rerun the DNS and HTTP check
 *
 * Out, one line each:
 *   EVT ...                      status
 *   CP fw=<a.b.c> target=<chip>  the co-processor
 *   AP ssid=".." rssi=.. ch=.. auth=..   one per access point, then
 *   EVT scan_done n=..
 *   NET dns|http ...             the end-to-end check
 *   STATUS ...                   state, IP, RSSI, free heap
 *   HB ...                       heartbeat every 10 s
 *   OK / ERR ...                 answers to commands
 */
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_hosted.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"
#include "nvs.h"
#include "nvs_flash.h"

#define NVS_NS        "wifi_test"
#define CMD_LINE_MAX  160
#define MAX_APS       32
#define HEARTBEAT_MS  10000
#define RETRY_MAX_MS  30000          /* reconnect backoff caps here */
#define NET_HOST      "example.com"

#define BIT_GOT_IP    BIT0
#define BIT_NET_CHECK BIT1
#define BIT_RETRY     BIT2

static esp_netif_t       *s_netif;
static EventGroupHandle_t s_events;
static volatile int       s_have_net;      /* a network is configured */
static volatile int       s_connected;     /* associated with the AP */
static volatile int       s_retry_ms;
static esp_timer_handle_t s_retry_timer;

static void out(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void out(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
}

/* ------------------------------------------------------------------ names */

static const char *auth_name(wifi_auth_mode_t m)
{
    switch (m) {
    case WIFI_AUTH_OPEN:            return "open";
    case WIFI_AUTH_WEP:             return "wep";
    case WIFI_AUTH_WPA_PSK:         return "wpa";
    case WIFI_AUTH_WPA2_PSK:        return "wpa2";
    case WIFI_AUTH_WPA_WPA2_PSK:    return "wpa/wpa2";
    case WIFI_AUTH_WPA3_PSK:        return "wpa3";
    case WIFI_AUTH_WPA2_WPA3_PSK:   return "wpa2/wpa3";
    case WIFI_AUTH_WPA2_ENTERPRISE: return "wpa2-ent";
    case WIFI_AUTH_OWE:             return "owe";
    default:                        return "other";
    }
}

/* ------------------------------------------------------------------- NVS */

static int creds_load(wifi_config_t *cfg)
{
    nvs_handle_t h;
    size_t       ns = sizeof(cfg->sta.ssid), np = sizeof(cfg->sta.password);

    memset(cfg, 0, sizeof(*cfg));
    if (nvs_open(NVS_NS, NVS_READONLY, &h) != ESP_OK)
        return 0;
    if (nvs_get_blob(h, "ssid", cfg->sta.ssid, &ns) != ESP_OK ||
        nvs_get_blob(h, "pass", cfg->sta.password, &np) != ESP_OK) {
        nvs_close(h);
        return 0;
    }
    nvs_close(h);
    return cfg->sta.ssid[0] != 0;
}

static esp_err_t creds_store(const wifi_config_t *cfg)
{
    nvs_handle_t h;
    esp_err_t    err = nvs_open(NVS_NS, NVS_READWRITE, &h);

    if (err != ESP_OK)
        return err;
    err = nvs_set_blob(h, "ssid", cfg->sta.ssid, sizeof(cfg->sta.ssid));
    if (err == ESP_OK)
        err = nvs_set_blob(h, "pass", cfg->sta.password, sizeof(cfg->sta.password));
    if (err == ESP_OK)
        err = nvs_commit(h);
    nvs_close(h);
    return err;
}

static void creds_erase(void)
{
    nvs_handle_t h;

    if (nvs_open(NVS_NS, NVS_READWRITE, &h) == ESP_OK) {
        nvs_erase_all(h);
        nvs_commit(h);
        nvs_close(h);
    }
}

/* ---------------------------------------------------------------- station */

/* Runs on the esp_timer task; the connect itself is an RPC to the C6, so it
 * is left to the main loop. */
static void retry_cb(void *arg)
{
    xEventGroupSetBits(s_events, BIT_RETRY);
}

static void on_wifi(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    switch (id) {
    case WIFI_EVENT_STA_START:
        out("EVT sta_start\n");
        break;
    case WIFI_EVENT_STA_CONNECTED: {
        wifi_event_sta_connected_t *e = data;
        s_connected = 1;
        s_retry_ms  = 1000;
        out("EVT connected ssid=\"%.*s\" ch=%d auth=%s\n", e->ssid_len,
            (const char *)e->ssid, e->channel, auth_name(e->authmode));
        break;
    }
    case WIFI_EVENT_STA_DISCONNECTED: {
        wifi_event_sta_disconnected_t *e = data;
        s_connected = 0;
        xEventGroupClearBits(s_events, BIT_GOT_IP);
        out("EVT disconnected reason=%d rssi=%d\n", e->reason, e->rssi);
        /* Retry with backoff while a network is configured. 201 (no AP
         * found) and 15/204 (bad password) keep retrying too: the AP may
         * just be out of range, and `forget` stops it. */
        if (s_have_net) {
            out("EVT retry_in ms=%d\n", s_retry_ms);
            esp_timer_stop(s_retry_timer);
            esp_timer_start_once(s_retry_timer, (uint64_t)s_retry_ms * 1000);
            s_retry_ms = s_retry_ms * 2 > RETRY_MAX_MS ? RETRY_MAX_MS : s_retry_ms * 2;
        }
        break;
    }
    default:
        break;
    }
}

static void on_ip(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = data;
        out("EVT got_ip ip=" IPSTR " mask=" IPSTR " gw=" IPSTR "\n",
            IP2STR(&e->ip_info.ip), IP2STR(&e->ip_info.netmask),
            IP2STR(&e->ip_info.gw));
        xEventGroupSetBits(s_events, BIT_GOT_IP | BIT_NET_CHECK);
    } else if (id == IP_EVENT_STA_LOST_IP) {
        out("EVT lost_ip\n");
    }
}

static void join(const wifi_config_t *stored)
{
    wifi_config_t cfg = *stored;

    /* An empty password means an open network; otherwise refuse anything
     * weaker than WPA2, and take WPA3 where the AP offers it. */
    cfg.sta.threshold.authmode = cfg.sta.password[0] ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;
    cfg.sta.sae_pwe_h2e        = WPA3_SAE_PWE_BOTH;
    cfg.sta.scan_method        = WIFI_ALL_CHANNEL_SCAN;
    cfg.sta.sort_method        = WIFI_CONNECT_AP_BY_SIGNAL;

    s_have_net = 1;
    s_retry_ms = 1000;
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &cfg));
    out("EVT joining ssid=\"%s\"\n", (const char *)cfg.sta.ssid);
    esp_wifi_connect();
}

static void leave(void)
{
    s_have_net = 0;
    esp_timer_stop(s_retry_timer);
    esp_wifi_disconnect();
}

/* ------------------------------------------------------------------ steps */

static void report_cp(void)
{
    esp_hosted_coprocessor_fwver_t v = { 0 };
    uint32_t chip = 0;
    char     target[16] = "?";

    if (esp_hosted_get_coprocessor_fwversion(&v) != ESP_OK) {
        out("ERR cp_version (no answer from the C6 over SDIO)\n");
        return;
    }
    esp_hosted_get_cp_info(&chip, target, sizeof(target));
    out("CP fw=%u.%u.%u target=%s chip_id=%u\n", (unsigned)v.major1,
        (unsigned)v.minor1, (unsigned)v.patch1, target, (unsigned)chip);
}

static void scan(void)
{
    static wifi_ap_record_t aps[MAX_APS];
    uint16_t                n = MAX_APS, total = 0, i;
    int64_t                 t0 = esp_timer_get_time();
    esp_err_t               err;

    out("EVT scanning\n");
    /* Blocking scan; the station keeps its connection meanwhile. */
    err = esp_wifi_scan_start(NULL, true);
    if (err != ESP_OK) {
        out("ERR scan %s\n", esp_err_to_name(err));
        return;
    }
    esp_wifi_scan_get_ap_num(&total);
    esp_wifi_scan_get_ap_records(&n, aps);
    for (i = 0; i < n; i++)
        out("AP ssid=\"%s\" rssi=%d ch=%d auth=%s bssid=" MACSTR "\n",
            (const char *)aps[i].ssid, aps[i].rssi, aps[i].primary,
            auth_name(aps[i].authmode), MAC2STR(aps[i].bssid));
    out("EVT scan_done n=%u shown=%u ms=%u\n", total, n,
        (unsigned)((esp_timer_get_time() - t0) / 1000));
}

/* A name lookup, then a plain-HTTP HEAD request: proves that DNS, and data
 * in both directions, make it through the C6 and the SDIO link. */
static void net_check(void)
{
    struct addrinfo  hints = { .ai_family = AF_INET, .ai_socktype = SOCK_STREAM };
    struct addrinfo *res = NULL;
    struct timeval   tv = { .tv_sec = 5 };
    char             buf[128], ip[16];
    int64_t          t0 = esp_timer_get_time();
    int              fd, err, got;
    static const char req[] = "HEAD / HTTP/1.0\r\nHost: " NET_HOST "\r\n\r\n";

    err = getaddrinfo(NET_HOST, "80", &hints, &res);
    if (err != 0 || !res) {
        out("NET dns host=%s ok=0 err=%d\n", NET_HOST, err);
        return;
    }
    inet_ntoa_r(((struct sockaddr_in *)res->ai_addr)->sin_addr, ip, sizeof(ip));
    out("NET dns host=%s ok=1 ip=%s ms=%u\n", NET_HOST, ip,
        (unsigned)((esp_timer_get_time() - t0) / 1000));

    t0 = esp_timer_get_time();
    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        out("NET http ok=0 err=socket\n");
        freeaddrinfo(res);
        return;
    }
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
        out("NET http ok=0 err=connect errno=%d\n", errno);
    } else if (send(fd, req, sizeof(req) - 1, 0) != (int)sizeof(req) - 1) {
        out("NET http ok=0 err=send errno=%d\n", errno);
    } else if ((got = recv(fd, buf, sizeof(buf) - 1, 0)) <= 0) {
        out("NET http ok=0 err=recv errno=%d\n", errno);
    } else {
        buf[got] = '\0';
        buf[strcspn(buf, "\r\n")] = '\0';           /* the status line */
        out("NET http ok=1 ms=%u status=\"%s\"\n",
            (unsigned)((esp_timer_get_time() - t0) / 1000), buf);
    }
    close(fd);
    freeaddrinfo(res);
}

static void status(void)
{
    esp_netif_ip_info_t ipi = { 0 };
    wifi_ap_record_t    ap;
    int                 rssi = 0;

    esp_netif_get_ip_info(s_netif, &ipi);
    if (s_connected && esp_wifi_sta_get_ap_info(&ap) == ESP_OK)
        rssi = ap.rssi;
    out("STATUS net=%s connected=%d ip=" IPSTR " rssi=%d heap=%u psram=%u\n",
        s_have_net ? "set" : "none", s_connected, IP2STR(&ipi.ip), rssi,
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

/* ---------------------------------------------------------------- console */

/* UART0 through the USB bridge, as whistle_stt. */
static void console_init(void)
{
    ESP_ERROR_CHECK(uart_driver_install(CONFIG_ESP_CONSOLE_UART_NUM, 1024, 0, 0, NULL, 0));
    uart_vfs_dev_use_driver(CONFIG_ESP_CONSOLE_UART_NUM);
    fcntl(fileno(stdin), F_SETFL, fcntl(fileno(stdin), F_GETFL) | O_NONBLOCK);
}

/* Next key=value pair from *p; the value may be "quoted". Returns 0 at the
 * end of the line. */
static int next_kv(char **p, char **key, char **val)
{
    char *s = *p;

    while (*s == ' ')
        s++;
    if (!*s)
        return 0;
    *key = s;
    while (*s && *s != '=' && *s != ' ')
        s++;
    if (*s != '=') {
        *val = s;                               /* key with no value */
    } else {
        *s++ = '\0';
        if (*s == '"') {
            *val = ++s;
            while (*s && *s != '"')
                s++;
        } else {
            *val = s;
            while (*s && *s != ' ')
                s++;
        }
    }
    if (*s)
        *s++ = '\0';
    *p = s;
    return 1;
}

static void cmd_join(char *args)
{
    wifi_config_t cfg = { 0 };
    char         *k, *v;
    esp_err_t     err;

    while (next_kv(&args, &k, &v)) {
        if (!strcmp(k, "ssid") && strlen(v) < sizeof(cfg.sta.ssid)) {
            strcpy((char *)cfg.sta.ssid, v);
        } else if (!strcmp(k, "pass") && strlen(v) < sizeof(cfg.sta.password)) {
            strcpy((char *)cfg.sta.password, v);
        } else {
            out("ERR join: bad argument %s (ssid: up to 32 bytes, pass: up to 63)\n", k);
            return;
        }
    }
    if (!cfg.sta.ssid[0]) {
        out("ERR join: ssid missing\n");
        return;
    }
    if (cfg.sta.password[0] && strlen((char *)cfg.sta.password) < 8) {
        out("ERR join: a WPA password has at least 8 characters\n");
        return;
    }
    err = creds_store(&cfg);
    if (err != ESP_OK) {
        out("ERR join: nvs %s\n", esp_err_to_name(err));
        return;
    }
    out("OK\n");
    leave();
    join(&cfg);
}

static void command(char *line)
{
    if (!strcmp(line, "scan")) {
        scan();
    } else if (!strncmp(line, "join ", 5)) {
        cmd_join(line + 5);
    } else if (!strcmp(line, "forget")) {
        leave();
        creds_erase();
        out("OK\n");
    } else if (!strcmp(line, "status")) {
        status();
    } else if (!strcmp(line, "net")) {
        if (xEventGroupGetBits(s_events) & BIT_GOT_IP)
            net_check();
        else
            out("ERR net: no IP yet\n");
    } else {
        out("ERR unknown command (scan, join ssid=.. pass=.., forget, status, net)\n");
    }
}

static void poll_console(void)
{
    static char line[CMD_LINE_MAX];
    static int  len, overflow;
    char        buf[64];
    int         got, i;

    while ((got = read(fileno(stdin), buf, sizeof(buf))) > 0) {
        for (i = 0; i < got; i++) {
            char c = buf[i];
            if (c == '\n' || c == '\r') {
                line[len] = '\0';
                if (overflow)
                    out("ERR line longer than %d bytes\n", CMD_LINE_MAX - 1);
                else if (len)
                    command(line);
                len = overflow = 0;
            } else if (len < CMD_LINE_MAX - 1) {
                line[len++] = c;
            } else {
                overflow = 1;
            }
        }
    }
}

/* ------------------------------------------------------------------- main */

void app_main(void)
{
    wifi_init_config_t   init = WIFI_INIT_CONFIG_DEFAULT();
    esp_timer_create_args_t targs = { .callback = retry_cb, .name = "wifi_retry" };
    wifi_config_t        stored;
    uint8_t              mac[6];
    int64_t              t_hb = 0, t0;
    esp_err_t            err;

    vTaskDelay(pdMS_TO_TICKS(500));
    console_init();
    out("EVT wifi_test booting\n");

    err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    s_events = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_timer_create(&targs, &s_retry_timer));
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_netif = esp_netif_create_default_wifi_sta();

    /* esp_hosted resets the C6 and brings the SDIO link up before app_main;
     * esp_wifi_init is the first call that has to cross it. */
    t0  = esp_timer_get_time();
    err = esp_wifi_init(&init);
    if (err != ESP_OK) {
        out("ERR wifi_init %s (no answer from the C6 over SDIO)\n", esp_err_to_name(err));
        return;
    }
    out("EVT wifi_init ms=%u\n", (unsigned)((esp_timer_get_time() - t0) / 1000));
    report_cp();

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_wifi, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, ESP_EVENT_ANY_ID, on_ip, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    if (esp_wifi_get_mac(WIFI_IF_STA, mac) == ESP_OK)
        out("EVT sta_mac " MACSTR "\n", MAC2STR(mac));

    scan();
    if (creds_load(&stored))
        join(&stored);
    else
        out("EVT no network stored: join ssid=<name> pass=<password>\n");
    out("EVT ready\n");

    for (;;) {
        poll_console();
        if (xEventGroupClearBits(s_events, BIT_NET_CHECK) & BIT_NET_CHECK)
            net_check();
        if ((xEventGroupClearBits(s_events, BIT_RETRY) & BIT_RETRY) &&
            s_have_net && !s_connected)
            esp_wifi_connect();
        if (esp_timer_get_time() - t_hb >= HEARTBEAT_MS * 1000) {
            t_hb = esp_timer_get_time();
            wifi_ap_record_t ap;
            out("HB connected=%d rssi=%d heap=%u\n", s_connected,
                s_connected && esp_wifi_sta_get_ap_info(&ap) == ESP_OK ? ap.rssi : 0,
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}
