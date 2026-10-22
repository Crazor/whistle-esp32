/* wh_run - run Whistle on the host, for rehearsal and validation.
 *
 *   wh_run model.cact clip.wav [--lang en] [--depth N]
 *          [--enc-out enc.f32] [--logits-out lg.f32 --tokens 2,8192,...]
 *
 * Prints the transcript and per-phase timings (always measured here; the
 * ND_PROFILE counters split them further when compiled in). The dump options
 * write raw little-endian float32 for tools/check_whistle.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nd_ledcmd.h"
#include "nd_quant.h"
#include "nd_whistle.h"

static const char *LANGS[] = { "en", "de", "fr", "es", "it", "nl", "pl" };
static const char *const COLORS6[] = { "red", "green", "blue", "yellow", "purple", "white" };

/* The LED demo's acceptance test for a transcript: it names a colour. */
static int has_command(const char *text, void *ctx)
{
    nd_led_cmd c;
    (void)ctx;
    return nd_led_parse(text, COLORS6, 6, 3.0f, &c);
}

static void *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    void *b;
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    *n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    b = malloc(*n);
    if (b && fread(b, 1, *n, f) != *n) {
        free(b);
        b = NULL;
    }
    fclose(f);
    return b;
}

static unsigned rd32(const unsigned char *p) { return p[0] | p[1] << 8 | p[2] << 16 | (unsigned)p[3] << 24; }
static unsigned rd16(const unsigned char *p) { return p[0] | p[1] << 8; }

/* 16 kHz mono 16-bit PCM WAV -> floats in [-1, 1]. */
static float *read_wav(const char *path, unsigned *count)
{
    size_t               n;
    unsigned char       *b = (unsigned char *)slurp(path, &n);
    size_t               off = 12;
    unsigned             rate = 0, ch = 0, bits = 0, i;
    float               *out = NULL;

    if (!b || n < 12 || memcmp(b, "RIFF", 4) || memcmp(b + 8, "WAVE", 4)) {
        fprintf(stderr, "not a RIFF/WAVE file: %s\n", path);
        free(b);
        return NULL;
    }
    while (off + 8 <= n) {
        unsigned len = rd32(b + off + 4);
        if (!memcmp(b + off, "fmt ", 4)) {
            ch = rd16(b + off + 10);
            rate = rd32(b + off + 12);
            bits = rd16(b + off + 22);
        } else if (!memcmp(b + off, "data", 4)) {
            if (rate != 16000 || ch != 1 || bits != 16) {
                fprintf(stderr, "want 16 kHz mono 16-bit, got %u Hz %u ch %u bit\n", rate, ch, bits);
                break;
            }
            if (len > n - off - 8)
                len = (unsigned)(n - off - 8);
            *count = len / 2;
            out = (float *)malloc(*count * sizeof(float));
            for (i = 0; i < *count; i++)
                out[i] = (short)rd16(b + off + 8 + 2 * i) / 32768.0f;
            break;
        }
        off += 8 + len + (len & 1);
    }
    free(b);
    return out;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int write_f32(const char *path, const float *x, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(x, sizeof(float), n, f) != n) {
        fprintf(stderr, "cannot write %s\n", path);
        if (f)
            fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv)
{
    const char *enc_out = NULL, *lg_out = NULL, *toks_arg = NULL, *kw = NULL;
    float       bonus = 0.0f, gate_db = 0.0f;
    int         protocol = 0, led_kw = 0, retry = -1;
    uint32_t    mask = 0x7F;
    float       margin = 0.2f;         /* the engine default */
    uint32_t    s0 = 0, s1 = 0;
    int         lang = 0, depth = 0, i, rc;
    size_t      size;
    void       *blob;
    unsigned    n;
    float      *x, *enc;
    nd_whistle  w;
    uint32_t    T, ids[320];
    char        text[4096];
    double      t0, t_enc, t_dec;

    if (argc < 3) {
        fprintf(stderr, "usage: %s model.cact clip.wav [--lang xx] [--depth N] "
                        "[--enc-out f] [--logits-out f --tokens a,b,...]\n", argv[0]);
        return 2;
    }
    for (i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--lang") && i + 1 < argc) {
            int k;
            for (k = 0, lang = -2; k < ND_WH_LANGS; k++)
                if (!strcmp(argv[i + 1], LANGS[k]))
                    lang = k;
            if (!strcmp(argv[i + 1], "auto"))
                lang = -1;                        /* detect, as the board does */
            i++;
        } else if (!strcmp(argv[i], "--led-keywords")) {
            led_kw = 1;                           /* the demo's 7-language lists */
        } else if (!strcmp(argv[i], "--margin") && i + 1 < argc) {
            margin = (float)atof(argv[++i]);      /* retry must score within this */
        } else if (!strcmp(argv[i], "--retry") && i + 1 < argc) {
            retry = atoi(argv[++i]);              /* languages re-tried on no command */
        } else if (!strcmp(argv[i], "--langs") && i + 1 < argc) {
            const char *l = argv[++i];            /* e.g. en,fr,es: detection set */
            int k;
            mask = 0;
            for (k = 0; k < ND_WH_LANGS; k++)
                if (strstr(l, LANGS[k]))
                    mask |= 1u << k;
        } else if (!strcmp(argv[i], "--depth") && i + 1 < argc) {
            depth = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--enc-out") && i + 1 < argc) {
            enc_out = argv[++i];
        } else if (!strcmp(argv[i], "--logits-out") && i + 1 < argc) {
            lg_out = argv[++i];
        } else if (!strcmp(argv[i], "--protocol")) {
            protocol = 1;      /* print the firmware's EVT/TXT/ACT lines */
        } else if (!strcmp(argv[i], "--float")) {
            nd_cq_int8 = 0;                   /* float matmuls (NumPy-exact) */
        } else if (!strcmp(argv[i], "--keywords") && i + 1 < argc) {
            kw = argv[++i];                   /* comma-separated */
        } else if (!strcmp(argv[i], "--bonus") && i + 1 < argc) {
            bonus = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--gate") && i + 1 < argc) {
            gate_db = (float)atof(argv[++i]); /* speech gate + trim, dB over floor */
        } else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) {
            toks_arg = argv[++i];
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    if (lang < -1) {
        fprintf(stderr, "language must be one of en de fr es it nl pl, or auto\n");
        return 2;
    }

    blob = slurp(argv[1], &size);
    if (!blob || (rc = nd_whistle_open(&w, blob, size)) != 0) {
        fprintf(stderr, "cannot open %s as Whistle (%d)\n", argv[1], blob ? rc : -1);
        return 1;
    }
    if (depth && nd_whistle_set_depth(&w, (uint32_t)depth) != 0) {
        fprintf(stderr, "depth must be 2..%u\n", w.dec_layers);
        return 2;
    }
    x = read_wav(argv[2], &n);
    if (!x)
        return 1;
    if (kw) {
        char *list = strdup(kw), *c;
        for (c = list; *c; c++)
            if (*c == ',')
                *c = '\n';
        fprintf(stderr, "keywords: %d token sequences, bonus %.1f\n",
                nd_whistle_set_keywords(&w, list, bonus ? bonus : 2.0f), bonus ? bonus : 2.0f);
        free(list);
    }
    if (protocol)
        printf("EVT ready model=whistle depth=%u enc_layers=%u (host)\nEVT recording\n"
               "EVT released held_ms=%u\n", w.depth, w.enc_layers, n / 16);
    w.lang_mask = mask;
    w.retry_margin = margin;
    if (retry >= 0)
        w.retry_langs = (uint32_t)retry;
    if (led_kw)
        w.accept = has_command;               /* as the board: retry when no colour */
    if (led_kw) {
        int k;
        nd_whistle_set_keywords(&w, NULL, 0.0f);
        w.kw_bonus = bonus ? bonus : 3.0f;
        for (k = 0; k < ND_WH_LANGS; k++)
            nd_whistle_add_keywords(&w, k, nd_led_keywords(k));
        fprintf(stderr, "keywords: %u token sequences in %d languages, bonus %.1f\n",
                w.n_kw, ND_WH_LANGS, w.kw_bonus);
    }
    if (gate_db > 0.0f) {
        if (!nd_whistle_find_speech(x, n, 0.0f, gate_db, 300, &s0, &s1)) {
            if (protocol) {
                printf("EVT no_speech\nTXT \nACT none\n");
                return 0;
            }
            printf("\n");
            fprintf(stderr, "no speech (gate %.0f dB)\n", gate_db);
            return 0;
        }
        fprintf(stderr, "speech %.2f-%.2f s of %.2f s\n", s0 / 16000.0, s1 / 16000.0, n / 16000.0);
        if (protocol)
            printf("EVT speech ms=%u frames=%u\n", (s1 - s0) / 16, nd_whistle_frames(&w, s1 - s0));
        memmove(x, x + s0, (size_t)(s1 - s0) * sizeof(float));
        n = s1 - s0;
    }

    T   = nd_whistle_frames(&w, n);
    enc = (float *)malloc((size_t)T * w.d * sizeof(float));
    t0  = now_ms();
    if (nd_whistle_encode(&w, x, n, enc) != 0) {
        fprintf(stderr, "clip too short\n");
        return 1;
    }
    t_enc = now_ms() - t0;
    if (enc_out && write_f32(enc_out, enc, (size_t)T * w.d))
        return 1;

    if (lg_out) {
        uint32_t toks[320], nt = 0;
        char    *s = toks_arg ? strdup(toks_arg) : NULL, *p;
        float   *lg;
        for (p = s ? strtok(s, ",") : NULL; p && nt < 320; p = strtok(NULL, ","))
            toks[nt++] = (uint32_t)strtoul(p, NULL, 10);
        free(s);
        lg = (float *)malloc((size_t)nt * w.vocab * sizeof(float));
        if (!nt || nd_whistle_logits(&w, enc, T, toks, nt, lg) != 0 ||
            write_f32(lg_out, lg, (size_t)nt * w.vocab))
            return 1;
        free(lg);
    }

    t0 = now_ms();
    rc = nd_whistle_decode(&w, enc, T, lang, ids, 320);
    t_dec = now_ms() - t0;
    if (rc < 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    nd_whistle_detok(&w, ids, (uint32_t)rc, text, sizeof(text));

    if (protocol) {
        static const char *const COLORS[] = { "red", "green", "blue", "yellow", "purple", "white" };
        nd_led_cmd cmd;
        printf("EVT lang %s\nTXT %s\nEVT time total_ms=%.0f encode_ms=%.0f decode_ms=%.0f tokens=%d audio_ms=%u\n",
               LANGS[w.lang], text, t_enc + t_dec, t_enc, t_dec, rc, n / 16);
#ifdef ND_PROFILE
        {
            static const char *PN[ND_WP_PHASES] = { "mel", "stem", "encoder", "cross_kv", "decoder" };
            for (i = 0; i < ND_WP_PHASES; i++)
                printf("EVT prof %-9s %8.0f ms\n", PN[i], nd_wprof[i] / 1000.0);
        }
#endif
        if (nd_led_parse(text, COLORS, 6, 3.0f, &cmd))
            printf("ACT led color=%s mode=%s duration=%.1f\n", COLORS[cmd.color],
                   cmd.flash ? "flash" : "solid", cmd.seconds);
        else
            printf("ACT none\n");
        nd_whistle_close(&w);
        return 0;
    }
    printf("%s\n", text);
    {   /* The LED demo's reading of it, as the firmware prints it. */
        static const char *const COLORS[] = { "red", "green", "blue", "yellow", "purple", "white" };
        nd_led_cmd cmd;
        if (nd_led_parse(text, COLORS, 6, 3.0f, &cmd))
            fprintf(stderr, "ACT led color=%s mode=%s duration=%.1f\n", COLORS[cmd.color],
                    cmd.flash ? "flash" : "solid", cmd.seconds);
        else
            fprintf(stderr, "ACT none\n");
    }
    fprintf(stderr, "tokens");
    for (i = 0; i < rc; i++)
        fprintf(stderr, " %u", ids[i]);
    if (lang < 0) {
        int k;
        fprintf(stderr, "\nlang_logits");
        for (k = 0; k < ND_WH_LANGS; k++)
            fprintf(stderr, " %s=%.2f", LANGS[k], w.lang_logit[k]);
    }
    fprintf(stderr, "\nscore %.3f first %.3f", w.score, w.score_first);
    fprintf(stderr, "\nlanguage %s%s\naudio %.2f s  frames %u  depth %u\n", LANGS[w.lang],
            lang < 0 ? " (detected)" : "", n / 16000.0, T, w.depth);
    fprintf(stderr, "time  encode %.0f ms  decode %.0f ms (%d tokens, %.1f ms/token)\n",
            t_enc, t_dec, rc, rc ? t_dec / (rc + 1) : 0.0);
#ifdef ND_PROFILE
    {
        static const char *PN[ND_WP_COUNT] = { "mel", "stem", "encoder", "cross_kv", "decoder",
                                               "~matmul", "~mlp", "~attn", "~mhc", "~head" };
        for (i = 0; i < ND_WP_COUNT; i++)
            fprintf(stderr, "prof  %-9s %8.1f ms\n", PN[i], nd_wprof[i] / 1000.0);
    }
#endif
    nd_whistle_close(&w);
    free(enc);
    free(x);
    free(blob);
    return 0;
}
