/* tts_run - Pico text to speech on the host, for rehearsal and timing.
 *
 *   tts_run lang_dir en "Flashing blue for five seconds." [out.wav]
 *
 * Writes 16 kHz mono 16-bit WAV (default tts.wav) and prints how long
 * synthesis took against the length of the speech. Reads the text from stdin
 * when it is "-".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nd_tts.h"

typedef struct {
    int16_t *pcm;
    size_t   n, cap;
} buf;

static int collect(const int16_t *pcm, size_t n, void *ctx)
{
    buf *b = (buf *)ctx;
    if (b->n + n > b->cap) {
        size_t   cap = b->cap ? b->cap * 2 : 1 << 16;
        int16_t *p;
        while (cap < b->n + n)
            cap *= 2;
        p = (int16_t *)realloc(b->pcm, cap * sizeof(*p));
        if (!p)
            return 1;
        b->pcm = p;
        b->cap = cap;
    }
    memcpy(b->pcm + b->n, pcm, n * sizeof(*pcm));
    b->n += n;
    return 0;
}

static void wr32(unsigned char *p, unsigned v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }
static void wr16(unsigned char *p, unsigned v) { p[0] = v; p[1] = v >> 8; }

static int write_wav(const char *path, const int16_t *pcm, size_t n)
{
    unsigned char h[44];
    FILE         *f = fopen(path, "wb");
    size_t        i;
    if (!f)
        return -1;
    memcpy(h, "RIFF", 4); wr32(h + 4, 36 + (unsigned)n * 2); memcpy(h + 8, "WAVEfmt ", 8);
    wr32(h + 16, 16); wr16(h + 20, 1); wr16(h + 22, 1); wr32(h + 24, ND_TTS_RATE);
    wr32(h + 28, ND_TTS_RATE * 2); wr16(h + 32, 2); wr16(h + 34, 16);
    memcpy(h + 36, "data", 4); wr32(h + 40, (unsigned)n * 2);
    fwrite(h, 1, sizeof(h), f);
    for (i = 0; i < n; i++) {           /* little-endian on any host */
        unsigned char s[2];
        wr16(s, (unsigned)(uint16_t)pcm[i]);
        fwrite(s, 1, 2, f);
    }
    return fclose(f);
}

static char *read_stdin(void)
{
    size_t n = 0, cap = 4096;
    char  *s = (char *)malloc(cap);
    size_t r;
    while (s && (r = fread(s + n, 1, cap - n - 1, stdin)) > 0) {
        n += r;
        if (n + 1 == cap)
            s = (char *)realloc(s, cap *= 2);
    }
    if (s)
        s[n] = 0;
    return s;
}

int main(int argc, char **argv)
{
    nd_tts          t;
    buf             b = { 0 };
    void           *mem;
    char           *text;
    const char     *out = argc > 4 ? argv[4] : "tts.wav";
    struct timespec t0, t1, t2;
    double          load_s, synth_s, speech_s;
    long            n;

    if (argc < 4) {
        fprintf(stderr, "usage: %s lang_dir lang text|- [out.wav]\n"
                        "  lang: en de fr es it (or en-US en-GB de-DE ...)\n", argv[0]);
        return 2;
    }
    text = strcmp(argv[3], "-") ? argv[3] : read_stdin();
    mem = malloc(ND_TTS_MEM_SIZE);
    if (!text || !mem)
        return 1;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (nd_tts_open(&t, mem, ND_TTS_MEM_SIZE, argv[1], argv[2])) {
        fprintf(stderr, "tts: %s\n", t.err);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    n = nd_tts_speak(&t, text, collect, &b);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    if (n < 0) {
        fprintf(stderr, "tts: %s\n", t.err);
        return 1;
    }
    nd_tts_close(&t);

    load_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    synth_s = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) * 1e-9;
    speech_s = (double)b.n / ND_TTS_RATE;
    printf("voice %s  load %.3f s  speech %.2f s  synth %.3f s  (%.1fx real time)\n",
           t.lang, load_s, speech_s, synth_s, synth_s > 0 ? speech_s / synth_s : 0.0);
    if (write_wav(out, b.pcm, b.n)) {
        fprintf(stderr, "tts: cannot write %s\n", out);
        return 1;
    }
    printf("wrote %s\n", out);
    free(b.pcm);
    free(mem);
    return 0;
}
