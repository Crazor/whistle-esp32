/* ledcmd_test - transcripts Whistle produced on the board, and what the LED
 * demo must make of them. */
#include <stdio.h>
#include <string.h>

#include "nd_ledcmd.h"

static const char *const COLORS[] = { "red", "green", "blue", "yellow", "purple", "white" };

static const struct { const char *text; int color, flash; float secs, period; } CASES[] = {
    { "flash delight red.",                         0, 1, 3.0f, 0.0f },
    { "Meg delight yellow for five seconds.",       3, 0, 5.0f, 0.0f },
    { "Make delight purple for two seconds.",       4, 0, 2.0f, 0.0f },
    { "Strobed the Blue Light every 0.5 seconds.",  2, 1, 3.0f, 0.5f },
    { "Make the light yellow for nine seconds.",    3, 0, 9.0f, 0.0f },
    { "Flash read for two seconds.",                0, 1, 2.0f, 0.0f },
    { "Blinking green, every second.",              1, 1, 3.0f, 1.0f },
    { "What is the capital of France?",            -1, 0, 3.0f, 0.0f },
    { "Turn it white for 3 seconds",                5, 0, 3.0f, 0.0f },
    /* French: real transcripts from the host engine, then phrasings */
    { "Il humine la lumière verte.",                1, 0, 3.0f, 0.0f },
    { "Mais la lumière en bleu pendant 5 secondes.", 2, 0, 5.0f, 0.0f },
    { "Fais clignoter la lumière rouge toutes les deux secondes.", 0, 1, 3.0f, 2.0f },
    { "Allume la lumière BLANCHE.",                 5, 0, 3.0f, 0.0f },
    /* German */
    { "Lass das Licht zwei Sekunden lang rot blinken.", 0, 1, 2.0f, 0.0f },
    { "Mach das Licht Weiß.",                       5, 0, 3.0f, 0.0f },
    { "Blau, alle 0,5 Sekunden.",                   2, 1, 3.0f, 0.5f },
    /* Spanish */
    { "Haz parpadear la luz roja durante dos segundos.", 0, 1, 2.0f, 0.0f },
    { "Pon la luz amarilla, por favor.",            3, 0, 3.0f, 0.0f },
    { "¿Verdad que es bonito?",                    -1, 0, 3.0f, 0.0f },
    /* Italian */
    { "Fai lampeggiare la luce gialla per due secondi.", 3, 1, 2.0f, 0.0f },
    { "Accendi la luce viola ogni secondo.",        4, 1, 3.0f, 1.0f },
    /* Dutch */
    { "Laat het licht twee seconden groen knipperen.", 1, 1, 2.0f, 0.0f },
    { "Zet de lamp op wit alsjeblieft.",            5, 0, 3.0f, 0.0f },
    /* Polish */
    { "Niech światło miga na czerwono przez dwie sekundy.", 0, 1, 2.0f, 0.0f },
    { "Zmień światło na ŻÓŁTE.",                    3, 0, 3.0f, 0.0f },
    { "Ustaw lampę na kolor fioletowy co trzy sekundy.", 4, 1, 3.0f, 3.0f },
    /* The README's examples */
    { "Make it red.",                               0, 0, 3.0f, 0.0f },
    { "Turn the light green for five seconds.",     1, 0, 5.0f, 0.0f },
    { "Flash the light blue.",                      2, 1, 3.0f, 0.0f },
    { "Blink purple every half a second.",          4, 1, 3.0f, 0.5f },
    { "Fais clignoter la lumière en violet.",       4, 1, 3.0f, 0.0f },
    { "Fais clignoter la lumière en blanc toutes les deux secondes.", 5, 1, 3.0f, 2.0f },
    { "Clignote en rouge toutes les demi-secondes.", 0, 1, 3.0f, 0.5f },
    { "Mets la LED en bleu.",                       2, 0, 3.0f, 0.0f },
    { "Lass das Licht rot blinken.",                0, 1, 3.0f, 0.0f },
    { "Blau blinken, alle zwei Sekunden.",          2, 1, 3.0f, 2.0f },
    { "Fai lampeggiare la luce verde.",             1, 1, 3.0f, 0.0f },
    { "Laat het licht blauw knipperen.",            2, 1, 3.0f, 0.0f },
    { "Niech światło miga na czerwono.",            0, 1, 3.0f, 0.0f },
};

int main(int argc, char **argv)
{
    int i, fails = 0;
    if (argc > 1 && !strcmp(argv[1], "--stdin")) {   /* parse transcripts, one per line */
        char line[1024];
        while (fgets(line, sizeof(line), stdin)) {
            nd_led_cmd c;
            if (nd_led_parse(line, COLORS, 6, 3.0f, &c))
                printf("ACT led color=%s mode=%s\n", COLORS[c.color], c.flash ? "flash" : "solid");
            else
                printf("ACT none\n");
        }
        return 0;
    }
    for (i = 0; i < (int)(sizeof(CASES) / sizeof(CASES[0])); i++) {
        nd_led_cmd c;
        int        found = nd_led_parse(CASES[i].text, COLORS, 6, 3.0f, &c);
        int        ok = c.color == CASES[i].color && (!found || (c.flash == CASES[i].flash &&
                        c.seconds == CASES[i].secs && c.period == CASES[i].period));
        printf("%s  %-44s -> %s %s %.1fs period %.1f\n", ok ? "ok  " : "FAIL", CASES[i].text,
               found ? COLORS[c.color] : "none", c.flash ? "flash" : "solid", c.seconds, c.period);
        fails += !ok;
    }
    printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
