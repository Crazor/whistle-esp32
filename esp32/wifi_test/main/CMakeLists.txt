# esp_hosted and esp_wifi_remote come from idf_component.yml.
idf_component_register(SRCS "main.c"
                       REQUIRES esp_wifi esp_netif esp_event nvs_flash lwip
                                esp_driver_uart esp_timer)
