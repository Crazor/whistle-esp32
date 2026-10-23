/* nd_comp.h - a speech compressor and limiter for a small speaker.
 *
 * Speech peaks far above its average level, so a quiet speaker plays it
 * quietly long before the peaks clip. This turns down the loud parts and
 * then turns everything up:
 *
 *   envelope   peak follower, 1 ms attack, 120 ms release
 *   compress   ratio ND_COMP_RATIO above ND_COMP_THRESH_DB
 *   makeup     ND_COMP_MAKEUP_DB
 *   limit      soft knee from 90% of `ceiling` up to `ceiling`, so the
 *              attack's few overshooting samples bend instead of clip
 *
 * In place on 16-bit mono PCM, single precision only.
 *
 * Tuned on Pico's voices with tools/check_tts.py --comp, which has Whistle
 * transcribe the result: about +4 dB of loudness over speech at the same peak
 * level, with transcripts as without it except one borderline Italian
 * phrase. Stronger settings (-24 dB, 8:1, +20 dB) gave +7 dB but doubled
 * the word error rate and lost two of ten commands.
 */
#ifndef ND_COMP_H
#define ND_COMP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ND_COMP_THRESH_DB
#define ND_COMP_THRESH_DB  -30.0f
#endif
#ifndef ND_COMP_RATIO
#define ND_COMP_RATIO      4.0f
#endif
#ifndef ND_COMP_MAKEUP_DB
#define ND_COMP_MAKEUP_DB  18.0f
#endif

typedef struct {
    float env;                  /* envelope, full scale = 1 */
    float att, rel;             /* per-sample smoothing coefficients */
    float thresh, slope;        /* threshold (linear), 1/ratio - 1 */
    float makeup;               /* linear */
    float ceiling;              /* limiter output peak, full scale = 1 */
} nd_comp;

/* `ceiling_db` is the highest output level, in dBFS (0 or below). Leave room
 * for gain applied after this, such as a codec's digital volume above 0 dB. */
void nd_comp_init(nd_comp *c, int rate, float ceiling_db);
void nd_comp_run(nd_comp *c, int16_t *pcm, size_t n);

#ifdef __cplusplus
}
#endif

#endif
