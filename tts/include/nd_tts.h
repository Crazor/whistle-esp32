/* nd_tts.h - text to speech with SVOX Pico (third_party/picotts).
 *
 * A thin wrapper over Pico's API: one voice at a time, text in as UTF-8,
 * 16 kHz mono 16-bit PCM out through a callback as it is produced.
 *
 * Pico speaks five of Whistle's seven languages. Each one is two lingware
 * files, a text analysis part (*_ta.bin) and a signal generation part
 * (*_sg.bin), read from `dir` at open and copied into the memory area:
 *
 *   en  en-US_ta.bin  en-US_lh0_sg.bin       (en-GB: en-GB_ta, en-GB_kh0_sg)
 *   de  de-DE_ta.bin  de-DE_gl0_sg.bin
 *   fr  fr-FR_ta.bin  fr-FR_nk0_sg.bin
 *   es  es-ES_ta.bin  es-ES_zl0_sg.bin
 *   it  it-IT_ta.bin  it-IT_cm0_sg.bin
 *
 * Dutch and Polish have no Pico voice; nd_tts_open() says so. tools/
 * pico_lang.py fetches the files.
 *
 * The memory area holds the lingware (0.9-1.5 MB per language) and the
 * engine's working state. ND_TTS_MEM_SIZE fits any one language.
 */
#ifndef ND_TTS_H
#define ND_TTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ND_TTS_RATE      16000
#define ND_TTS_MEM_SIZE  (3u << 20)

typedef struct {
    void *sys, *engine, *ta, *sg;      /* Pico handles */
    char  lang[8];                     /* e.g. "en-US" */
    char  err[200];                    /* last error, for the log */
} nd_tts;

/* Return nonzero to stop speaking. */
typedef int (*nd_tts_sink)(const int16_t *pcm, size_t n, void *ctx);

/* `lang` is a Whistle code ("en", "de", "fr", "es", "it") or a Pico locale
 * ("en-US", "en-GB", ...). `mem` must stay valid until nd_tts_close().
 * Returns 0, or -1 with the reason in t->err. */
int  nd_tts_open(nd_tts *t, void *mem, size_t mem_size, const char *dir, const char *lang);

/* Synthesise `text` (UTF-8) and hand the samples to `sink` in chunks of up
 * to a few hundred. Pico's <speed level="..">, <pitch level="..">, and
 * <volume level=".."> markup works inline, with 100 as normal. Returns the
 * number of samples produced, or -1 with the reason in t->err. */
long nd_tts_speak(nd_tts *t, const char *text, nd_tts_sink sink, void *ctx);

void nd_tts_close(nd_tts *t);

/* The Pico locale for a Whistle language code, or NULL if Pico has none. */
const char *nd_tts_locale(const char *lang);

#ifdef __cplusplus
}
#endif

#endif
