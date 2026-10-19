/* nd_ledcmd.h - turn a Whistle transcript into an LED command.
 *
 * Whistle gives text, not a tool call, so the LED demo maps the transcript
 * onto the set_led tool's arguments (tools/led.json) directly: a colour from
 * the caller's (English) list, flash or solid, a duration and a blink period.
 *
 * One table covers Whistle's seven languages (en de fr es it nl pl), so the
 * parser need not know which one was spoken. Colours and flash verbs match by
 * word or by stem, which covers inflection ("verte", "rojas", "czerwonym",
 * "strobed", "blinkt"); a few homophones Whistle wrote in testing fold onto
 * colours ("read" -> red, "blew" -> blue). If several colours are named, the
 * last one wins. "every / toutes les / alle / cada / ogni / elke / co N
 * seconds" is a blink period: it means flash, not a duration.
 */
#ifndef ND_LEDCMD_H
#define ND_LEDCMD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int   color;      /* index into the caller's colour list, or -1 */
    int   flash;      /* 1 for flash / blink, 0 for solid */
    float seconds;    /* stated duration, or the default */
    float period;     /* "every N seconds": one on/off cycle, or 0 */
} nd_led_cmd;

/* Returns 1 if a colour was found. `colors` are English names (red, green,
 * blue, yellow, purple, white); `seconds` defaults to default_s. */
int nd_led_parse(const char *text, const char *const *colors, int n_colors,
                 float default_s, nd_led_cmd *out);

/* Words worth biasing Whistle towards for language `lang` (index into
 * "en de fr es it nl pl"), newline-separated: the colours in their common
 * forms and the blink verbs. NULL for an unknown index. */
const char *nd_led_keywords(int lang);

/* Lowercase UTF-8 in place: ASCII, Latin-1 and Latin Extended-A capitals
 * (À-Þ, Ą Ć Ę Ł Ń Ś Ź Ż ...), keeping the byte length. */
void nd_utf8_lower(char *s);

#ifdef __cplusplus
}
#endif
#endif /* ND_LEDCMD_H */
