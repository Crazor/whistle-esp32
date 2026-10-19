#include "nd_ledcmd.h"

#include <stdlib.h>
#include <string.h>

/* word -> canonical colour. '*' at the end matches as a stem. Short words
 * stay exact so they do not catch others ("rot" vs "rotate", "verde" vs
 * Spanish "verdad"). */
static const struct { const char *w, *color; } COLOURS[] = {
    /* en */ { "red", "red" }, { "read", "red" }, { "green", "green" }, { "blue", "blue" },
             { "blew", "blue" }, { "yellow", "yellow" }, { "purple", "purple" }, { "white", "white" },
    /* de */ { "rot", "red" }, { "rote*", "red" }, { "grün*", "green" }, { "gruen*", "green" },
             { "blau*", "blue" }, { "gelb*", "yellow" }, { "lila", "purple" }, { "violett*", "purple" },
             { "weiß*", "white" }, { "weiss*", "white" },
    /* fr */ { "roug*", "red" }, { "vert", "green" }, { "verte*", "green" }, { "verts", "green" },
             { "bleu*", "blue" }, { "jaune*", "yellow" }, { "violet*", "purple" }, { "blanc*", "white" },
    /* es */ { "roj*", "red" }, { "verde", "green" }, { "verdes", "green" }, { "azul*", "blue" },
             { "amarill*", "yellow" }, { "morad*", "purple" }, { "púrpura", "purple" }, { "blanc*", "white" },
    /* it */ { "ross*", "red" }, { "verdi", "green" }, { "blu", "blue" }, { "giall*", "yellow" },
             { "viola", "purple" }, { "porpora", "purple" }, { "bianc*", "white" },
    /* nl */ { "rood", "red" }, { "rode", "red" }, { "groen*", "green" }, /* blauw: "blau*" */
             { "geel", "yellow" }, { "gele", "yellow" }, { "paars*", "purple" }, { "wit", "white" },
             { "witte", "white" },
    /* pl */ { "czerwon*", "red" }, { "zielon*", "green" }, { "niebiesk*", "blue" },
             { "żółt*", "yellow" }, { "zolt*", "yellow" }, { "fiolet*", "purple" },
             { "biał*", "white" }, { "bial*", "white" },
};

/* Blink verbs and nouns, all stems: "flashing", "blinkt", "clignote",
 * "parpadea", "lampeggia", "knippert", "miga". */
static const char *const FLASH[] = {
    "flash", "blink", "strob", "flicker", "puls",          /* en (de: blink) */
    "flacker",                                             /* de */
    "clignot", "scintill",                                 /* fr, it */
    "parpade", "destell", "intermit",                      /* es, it */
    "lampegg", "sfarfall",                                 /* it */
    "knipper", "flits",                                    /* nl */
    "miga", "mrug", "błysk", "blysk",                      /* pl */
};

/* "second(s)" in every language, as stems: second/seconde/secondi/secondes,
 * Sekunde/sekundy, segundo. */
static const char *const SECOND[] = { "second", "sekund", "segund" };

/* "every": every, jede/alle, toutes/chaque, cada, ogni, elke/iedere, co. */
static const char *const EVERY[] = {
    "every", "jede", "jeden", "jedes", "alle", "toutes", "tous", "chaque", "cada",
    "ogni", "elke", "iedere", "co",
};

static const struct { const char *w; float v; } NUMBERS[] = {
    { "a", 1 }, { "one", 1 }, { "two", 2 }, { "three", 3 }, { "four", 4 }, { "five", 5 },
    { "six", 6 }, { "seven", 7 }, { "eight", 8 }, { "nine", 9 }, { "ten", 10 }, { "half", 0.5f },
    { "ein", 1 }, { "eine", 1 }, { "eins", 1 }, { "zwei", 2 }, { "drei", 3 }, { "vier", 4 },
    { "fünf", 5 }, { "funf", 5 }, { "sechs", 6 }, { "sieben", 7 }, { "acht", 8 }, { "neun", 9 },
    { "zehn", 10 }, { "halbe", 0.5f },
    { "un", 1 }, { "une", 1 }, { "deux", 2 }, { "trois", 3 }, { "quatre", 4 }, { "cinq", 5 },
    { "sept", 7 }, { "huit", 8 }, { "neuf", 9 }, { "dix", 10 }, { "demi", 0.5f }, { "demie", 0.5f },
    { "uno", 1 }, { "una", 1 }, { "dos", 2 }, { "tres", 3 }, { "cuatro", 4 }, { "cinco", 5 },
    { "seis", 6 }, { "siete", 7 }, { "ocho", 8 }, { "nueve", 9 }, { "diez", 10 }, { "medio", 0.5f },
    { "due", 2 }, { "tre", 3 }, { "quattro", 4 }, { "cinque", 5 }, { "sei", 6 }, { "sette", 7 },
    { "otto", 8 }, { "nove", 9 }, { "dieci", 10 }, { "mezzo", 0.5f },
    { "een", 1 }, { "één", 1 }, { "twee", 2 }, { "drie", 3 }, { "vijf", 5 }, { "zes", 6 },
    { "zeven", 7 }, { "negen", 9 }, { "tien", 10 }, { "halve", 0.5f },
    { "jeden", 1 }, { "jedna", 1 }, { "jedną", 1 }, { "dwie", 2 }, { "dwa", 2 }, { "trzy", 3 },
    { "cztery", 4 }, { "pięć", 5 }, { "piec", 5 }, { "sześć", 6 }, { "szesc", 6 }, { "siedem", 7 },
    { "osiem", 8 }, { "dziewięć", 9 }, { "dziesięć", 10 }, { "pół", 0.5f },
};

static const char *const KEYWORDS[] = {
    /* en */ "red\ngreen\nblue\nyellow\npurple\nwhite\nflash\nblink",
    /* de */ "rot\ngrün\nblau\ngelb\nlila\nweiß\nblinken\nblinkt",
    /* fr */ "rouge\nvert\nverte\nbleu\nbleue\njaune\nviolet\nviolette\nblanc\nblanche\nclignoter\nclignote",
    /* es */ "rojo\nroja\nverde\nazul\namarillo\namarilla\nmorado\nmorada\nblanco\nblanca\nparpadear\nparpadea",
    /* it */ "rosso\nrossa\nverde\nblu\ngiallo\ngialla\nviola\nbianco\nbianca\nlampeggiare\nlampeggia",
    /* nl */ "rood\ngroen\nblauw\ngeel\npaars\nwit\nknipperen\nknippert",
    /* pl */ ("czerwony\nczerwone\nzielony\nzielone\nniebieski\nniebieskie\nżółty\nżółte\n"
              "fioletowy\nfioletowe\nbiały\nbiałe\nmiga"),
};

const char *nd_led_keywords(int lang)
{
    return lang >= 0 && lang < (int)(sizeof(KEYWORDS) / sizeof(KEYWORDS[0])) ? KEYWORDS[lang] : NULL;
}

void nd_utf8_lower(char *s)
{
    unsigned char *p = (unsigned char *)s;
    while (*p) {
        if (*p >= 'A' && *p <= 'Z') {
            *p += 32;
        } else if (p[0] == 0xC3 && p[1] >= 0x80 && p[1] <= 0x9E && p[1] != 0x97) {
            p[1] += 0x20;                          /* À..Þ -> à..þ */
            p++;
        } else if ((p[0] == 0xC4 || p[0] == 0xC5) && p[1] >= 0x80 && p[1] <= 0xBF) {
            /* Latin Extended-A: capitals are even code points, except in
             * U+0139-0148 (Ĺ..Ň, with Ł Ń) and U+0179-017E (Ź Ż Ž): odd. */
            unsigned cp = ((p[0] & 0x1Fu) << 6) | (p[1] & 0x3Fu);
            int upper = ((cp >= 0x139 && cp <= 0x148) || (cp >= 0x179 && cp <= 0x17E))
                        ? (int)(cp & 1u) : !(cp & 1u);
            if (upper && cp != 0x138 && cp != 0x149) {
                cp++;
                p[0] = (unsigned char)(0xC0 | (cp >> 6));
                p[1] = (unsigned char)(0x80 | (cp & 0x3F));
            }
            p++;
        }
        if (*p)
            p++;
    }
}

static int is_letter(unsigned char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c >= 0x80;
}

/* Next word, lowercased; "0,5" and "0.5" stay one word (as "0.5"). */
static int next_word(const char **p, char *w, int cap)
{
    const unsigned char *s = (const unsigned char *)*p;
    int                  n = 0;
    while (*s && !is_letter(*s))
        s++;
    while (*s && (is_letter(*s) ||
                  ((*s == '.' || *s == ',') && n && w[n - 1] >= '0' && w[n - 1] <= '9' &&
                   s[1] >= '0' && s[1] <= '9'))) {
        if (n < cap - 1)
            w[n++] = (char)(*s == ',' ? '.' : *s);
        s++;
    }
    w[n] = '\0';
    *p = (const char *)s;
    nd_utf8_lower(w);
    return n;
}

static int match(const char *word, const char *pat)
{
    size_t n = strlen(pat);
    if (n && pat[n - 1] == '*')
        return !strncmp(word, pat, n - 1);
    return !strcmp(word, pat);
}

static int in_list(const char *w, const char *const *list, int n, int stem)
{
    int i;
    for (i = 0; i < n; i++)
        if (stem ? !strncmp(w, list[i], strlen(list[i])) : !strcmp(w, list[i]))
            return 1;
    return 0;
}

static float number(const char *w)
{
    int i;
    if (w[0] >= '0' && w[0] <= '9')
        return (float)atof(w);
    for (i = 0; i < (int)(sizeof(NUMBERS) / sizeof(NUMBERS[0])); i++)
        if (!strcmp(w, NUMBERS[i].w))
            return NUMBERS[i].v;
    return -1.0f;
}

#define N(a) ((int)(sizeof(a) / sizeof((a)[0])))

int nd_led_parse(const char *text, const char *const *colors, int n_colors,
                 float default_s, nd_led_cmd *out)
{
    const char *p = text;
    char        w[40], h[3][40] = { "", "", "" };   /* h[0] = previous word */
    int         i, k;

    out->color   = -1;
    out->flash   = 0;
    out->seconds = default_s;
    out->period  = 0.0f;
    while (next_word(&p, w, sizeof(w))) {
        for (i = 0; i < N(COLOURS); i++)
            if (match(w, COLOURS[i].w))
                for (k = 0; k < n_colors; k++)
                    if (!strcmp(COLOURS[i].color, colors[k]))
                        out->color = k;
        if (in_list(w, FLASH, N(FLASH), 1))
            out->flash = 1;
        if (in_list(w, SECOND, N(SECOND), 1)) {
            int   every = 0;
            float v = number(h[0]);
            if (!strcmp(h[0], "a") && !strcmp(h[1], "half"))
                v = 0.5f;                              /* "half a second" */
            for (k = 0; k < 3; k++)
                every |= in_list(h[k], EVERY, N(EVERY), 0);
            if (every) {
                out->period = v > 0.0f && v <= 10.0f ? v : 1.0f;   /* "every second" */
                out->flash  = 1;
            } else if (v > 0.0f && v <= 30.0f) {     /* led.json: maximum 30 */
                out->seconds = v;
            }
        }
        memcpy(h[2], h[1], sizeof(h[2]));
        memcpy(h[1], h[0], sizeof(h[1]));
        memcpy(h[0], w, sizeof(h[0]));
    }
    return out->color >= 0;
}
