/* nd_tts.c - see nd_tts.h. */
#include "nd_tts.h"

#include <stdio.h>
#include <string.h>

#include "picoapi.h"

#define VOICE     "nd"
#define OUT_BYTES 256    /* pico_getData returns at most 128 samples a call */

static const struct {
    const char *code, *locale, *ta, *sg;
} VOICES[] = {
    { "en", "en-US", "en-US_ta.bin", "en-US_lh0_sg.bin" },
    { NULL, "en-GB", "en-GB_ta.bin", "en-GB_kh0_sg.bin" },
    { "de", "de-DE", "de-DE_ta.bin", "de-DE_gl0_sg.bin" },
    { "fr", "fr-FR", "fr-FR_ta.bin", "fr-FR_nk0_sg.bin" },
    { "es", "es-ES", "es-ES_ta.bin", "es-ES_zl0_sg.bin" },
    { "it", "it-IT", "it-IT_ta.bin", "it-IT_cm0_sg.bin" },
};
#define N_VOICES (sizeof(VOICES) / sizeof(VOICES[0]))

static int find_voice(const char *lang)
{
    unsigned i;
    for (i = 0; i < N_VOICES; i++)
        if ((VOICES[i].code && !strcmp(lang, VOICES[i].code)) || !strcmp(lang, VOICES[i].locale))
            return (int)i;
    return -1;
}

const char *nd_tts_locale(const char *lang)
{
    int v = find_voice(lang);
    return v < 0 ? NULL : VOICES[v].locale;
}

/* Record why `what` failed. Pico's own message when it has one. */
static int fail(nd_tts *t, const char *what, pico_Status st)
{
    pico_Retstring msg = "";
    if (t->engine)
        pico_getEngineStatusMessage((pico_Engine)t->engine, st, msg);
    else if (t->sys)
        pico_getSystemStatusMessage((pico_System)t->sys, st, msg);
    snprintf(t->err, sizeof(t->err), "%s: %s (%d)", what, msg, (int)st);
    return -1;
}

static int load(nd_tts *t, const char *dir, const char *file, void **res)
{
    char           path[PICO_MAX_DATAPATH_NAME_SIZE + PICO_MAX_FILE_NAME_SIZE];
    pico_Retstring name;
    pico_Status    st;
    pico_System    sys = (pico_System)t->sys;

    snprintf(path, sizeof(path), "%s/%s", dir, file);
    st = pico_loadResource(sys, (const pico_Char *)path, (pico_Resource *)res);
    if (st)
        return fail(t, path, st);
    st = pico_getResourceName(sys, (pico_Resource)*res, name);
    if (!st)
        st = pico_addResourceToVoiceDefinition(sys, (const pico_Char *)VOICE, (const pico_Char *)name);
    return st ? fail(t, file, st) : 0;
}

int nd_tts_open(nd_tts *t, void *mem, size_t mem_size, const char *dir, const char *lang)
{
    pico_Status st;
    int         v = find_voice(lang);

    memset(t, 0, sizeof(*t));
    if (v < 0) {
        snprintf(t->err, sizeof(t->err), "no Pico voice for '%s' (en, de, fr, es, it, en-GB)", lang);
        return -1;
    }
    strcpy(t->lang, VOICES[v].locale);

    st = pico_initialize(mem, (pico_Uint32)mem_size, (pico_System *)&t->sys);
    if (st) {
        t->sys = NULL;
        return fail(t, "pico_initialize", st);
    }
    st = pico_createVoiceDefinition((pico_System)t->sys, (const pico_Char *)VOICE);
    if (st || load(t, dir, VOICES[v].ta, &t->ta) || load(t, dir, VOICES[v].sg, &t->sg)
        || (st = pico_newEngine((pico_System)t->sys, (const pico_Char *)VOICE, (pico_Engine *)&t->engine))) {
        if (st)
            fail(t, "pico voice", st);
        t->engine = NULL;
        nd_tts_close(t);
        return -1;
    }
    return 0;
}

long nd_tts_speak(nd_tts *t, const char *text, nd_tts_sink sink, void *ctx)
{
    pico_Engine     e = (pico_Engine)t->engine;
    const pico_Char *in = (const pico_Char *)text;
    size_t          left = strlen(text) + 1;    /* the '\0' ends the utterance */
    int16_t         out[OUT_BYTES / 2];
    long            total = 0;

    while (left) {
        pico_Int16  sent = 0, got, type;
        pico_Status st = pico_putTextUtf8(e, in, (pico_Int16)(left > 32767 ? 32767 : left), &sent);
        if (st)
            return fail(t, "pico_putTextUtf8", st);
        in += sent;
        left -= (size_t)sent;

        /* Drain everything the text put so far produces. The input buffer
         * is small, so long text goes in over several rounds. */
        do {
            st = pico_getData(e, out, OUT_BYTES, &got, &type);
            if (got > 0) {
                total += got / 2;
                if (sink && sink(out, (size_t)got / 2, ctx)) {
                    pico_resetEngine(e, PICO_RESET_SOFT);
                    return total;
                }
            }
        } while (st == PICO_STEP_BUSY);
        if (st != PICO_STEP_IDLE) {
            fail(t, "pico_getData", st);
            pico_resetEngine(e, PICO_RESET_FULL);
            return -1;
        }
    }
    return total;
}

void nd_tts_close(nd_tts *t)
{
    pico_System sys = (pico_System)t->sys;
    if (!sys)
        return;
    if (t->engine)
        pico_disposeEngine(sys, (pico_Engine *)&t->engine);
    pico_releaseVoiceDefinition(sys, (const pico_Char *)VOICE);
    if (t->sg)
        pico_unloadResource(sys, (pico_Resource *)&t->sg);
    if (t->ta)
        pico_unloadResource(sys, (pico_Resource *)&t->ta);
    pico_terminate((pico_System *)&t->sys);
    t->sys = t->engine = t->ta = t->sg = NULL;
}
