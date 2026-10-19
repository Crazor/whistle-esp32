/* nd_whistle.h - Whistle speech recognition, read from a `.cact` v4 blob.
 *
 *   audio --log-mel--> [frames, 80] --stem--> [frames/8, 512]
 *         --encoder (8 non-causal blocks, mHC lanes)--> [frames/8, 512]
 *         --decoder (8 causal blocks + gated cross attention)--> tokens
 *
 * The encoder block is a Conformer-shaped Simple Attention block: half-step
 * Hadamard MLP, RoPE GQA attention, conv module (pw 512->1024, GLU, 9-tap
 * depthwise, norm, SiLU, pw 512->512), half-step Hadamard MLP. The decoder
 * block is Needle 3's (engram, Q/K/V causal 3-tap conv, gated GQA, Monarch
 * Hadamard MLP) with a gated cross attention after self attention:
 *
 *     x <- x + sigmoid(g) * post_norm(softmax(q K^T / sqrt(d)) V)
 *
 * Neither model has published source; tools/whistle_ref.py is the NumPy
 * reference this mirrors, and tools/check_whistle.py holds both to the
 * official engine. The weights are used in place (mmap'd flash on the ESP32);
 * activations go through ND_ALLOC like the Needle 2 engine.
 */
#ifndef ND_WHISTLE_H
#define ND_WHISTLE_H

#include <stddef.h>
#include <stdint.h>

#include "nd_cact.h"
#include "nd_model.h"       /* ND_ALLOC / ND_FREE / ND_ALLOC_FAST */
#include "nd_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ND_WH_LANGS       7
#define ND_WH_MAX_LAYERS  16
#define ND_WH_MAX_KW      192    /* keyword token sequences for biasing */
#define ND_WH_KW_TOKENS   8      /* tokens per phrase */

/* Per-phase microsecond counters, filled when ND_PROFILE is compiled in.
 * The first five are the pipeline's phases; the rest break them down and
 * overlap them (every phase does matmuls). */
enum { ND_WP_MEL, ND_WP_STEM, ND_WP_ENCODER, ND_WP_CROSS_KV, ND_WP_DECODER,
       ND_WP_LIN, ND_WP_MLP, ND_WP_ATTN, ND_WP_MHC, ND_WP_COUNT };
#define ND_WP_PHASES 5
extern uint64_t nd_wprof[ND_WP_COUNT];

typedef struct {
    nd_cact       c;
    nd_tokenizer  tok;
    nd_tensor    *t;            /* every directory record, decoded once */
    const void  **p;            /* and its payload pointer */

    /* geometry */
    uint32_t      d, nh, nkv, qk, vd, lanes, taps, vocab, max_seq;
    uint32_t      dec_layers, enc_layers;
    uint32_t      n_sites, sites[4];
    uint32_t      mels, n_fft, win, hop;
    float         rope_theta;
    float         rope_inv[32]; /* qk_head_dim / 2 inverse frequencies */
    float         fft_cos[256], fft_sin[256];   /* twiddles for the 512 FFT */
    float         hann[512];    /* periodic Hann over `win`, centred in n_fft */
    uint16_t      fb_lo[128], fb_hi[128];       /* each band's non-zero bins */

    /* canon positions (see tools/whistle_cact.py SCHEMA) */
    uint32_t      i_emb, i_dec, i_dec_mhc, i_perm1, i_perm2, i_engram;
    uint32_t      i_dec_final, i_audio_cfg, i_cross, i_pe_gate, i_enc;
    uint32_t      i_enc_mhc, i_enc_final, i_stem, i_melfb, i_tok;

    /* decoder depth (ladder): active layers, ascending */
    uint32_t      depth, active[ND_WH_MAX_LAYERS];

    /* keyword biasing: token sequences favoured during decoding */
    uint32_t      n_kw, n_groups, kw_len[ND_WH_MAX_KW];
    uint32_t      kw[ND_WH_MAX_KW][ND_WH_KW_TOKENS];
    uint16_t      kw_group[ND_WH_MAX_KW];   /* spellings of one phrase */
    int8_t        kw_lang[ND_WH_MAX_KW];    /* language it applies to, -1 = all */
    char          kw_text[ND_WH_MAX_KW][24]; /* each group's phrase, lowercase */
    float         kw_bonus;
    int           lang;         /* language of the last decode (detected or forced) */
    float         lang_logit[ND_WH_LANGS];   /* the detection's scores */
    float         score;        /* mean log-prob of the last transcript (unbiased) */
    float         score_first;  /* ... of the first pass, before any retry */
    float         retry_margin; /* a retry must score >= score_first - this */

    /* Detection on a short command is often confidently wrong (Dutch read as
     * English). When `accept` is set and rejects the first transcript (the
     * LED demo: "no colour in it"), the next `retry_langs` languages by
     * detection score are decoded in turn and the first transcript `accept`
     * likes replaces it; if none does, the first transcript stands. `lang_mask`
     * (bit i = language i, default all) limits detection to some languages. */
    int         (*accept)(const char *text, void *ctx);
    void         *accept_ctx;
    uint32_t      retry_langs;
    uint32_t      lang_mask;

    /* decoder session (allocated by nd_whistle_decode) */
    uint32_t      enc_T;
    float        *cross_k;      /* [layer][nh][T][qk], k_norm applied */
    float        *cross_v;      /* [layer][nh][T][vd] */
    float        *k_cache;      /* [layer][max_seq][nkv*qk] */
    float        *v_cache;      /* [layer][max_seq][nkv*vd] */
    float        *qkv_hist;     /* [layer][taps-1][q|k|v raw] */
    float        *eg_hist;      /* [site][16][d] raw engram v */
    uint32_t      hist[8];      /* recent token ids */
    float        *lane, *lane2, *xh, *lut, *tmp;
} nd_whistle;

/* Open over a whistle.cact blob used in place. Returns 0 on success, negative
 * on a blob that is not v4 or does not have Whistle's tensor layout. */
int  nd_whistle_open(nd_whistle *w, const void *blob, size_t size);
void nd_whistle_close(nd_whistle *w);

/* Use only `depth` decoder layers (2..dec_layers), the nested ladder subset
 * --audio-depth selects. Returns 0, or -1 for an out-of-range depth. */
int  nd_whistle_set_depth(nd_whistle *w, uint32_t depth);

/* Copy the decoder's hot weights (the tied 8199-row output head, read in
 * full every token) out of the mapped blob into `alloc`ed memory - on the
 * ESP32-S3, octal PSRAM, which streams faster than the QIO flash mapping.
 * Returns the bytes copied, or 0. */
size_t nd_whistle_cache_head(nd_whistle *w, void *(*alloc)(size_t));

/* Number of encoder frames for `n` samples: ceil^3(n / hop / 2). */
uint32_t nd_whistle_frames(const nd_whistle *w, uint32_t n);

/* Log-mel front end: `n` samples in [-1, 1] -> out[n/hop][mels]. */
void nd_whistle_logmel(const nd_whistle *w, const float *x, uint32_t n, float *out);

/* Front end + stem + encoder: out[nd_whistle_frames(n)][d]. What the
 * official engine's needle_embed returns. Returns 0 on success. */
int  nd_whistle_encode(nd_whistle *w, const float *x, uint32_t n, float *out);

/* Keyword biasing: newline-separated words or phrases, favoured by `bonus`
 * nats on the next token of any phrase the transcript has started (and on a
 * phrase's first token). Each phrase is added as written and capitalised.
 * NULL or bonus 0 clears. Returns the number of token sequences stored. */
int  nd_whistle_set_keywords(nd_whistle *w, const char *list, float bonus);

/* Add phrases that only apply when the decode's language is `lang` (index
 * into "en de fr es it nl pl"), so French colour words are not pushed into an
 * English transcript. Keeps what is already set. Returns sequences stored. */
int  nd_whistle_add_keywords(nd_whistle *w, int lang, const char *list);

/* Speech gate. Whistle hallucinates on a clip of microphone noise (the
 * official engine returns "Thank you." for 10 s of a quiet INMP441; its own
 * silence check only catches digital zeros). Returns 1 and the sample range
 * [*s0, *s1) to transcribe - the loud part plus `margin_ms` either side - if
 * at least 100 ms of 20 ms windows sit `min_db` above `floor_rms`, else 0.
 * floor_rms <= 0 estimates the floor from the clip (10th-percentile window),
 * which needs some quiet in the clip; a recorder that runs between presses
 * knows its floor better and should pass it. */
int  nd_whistle_find_speech(const float *x, uint32_t n, float floor_rms,
                            float min_db, uint32_t margin_ms,
                            uint32_t *s0, uint32_t *s1);

/* Greedy decode over an encoder output. `lang` forces a language (index into
 * "en de fr es it nl pl"); -1 detects it the way the official engine does:
 * the decoder's own choice among the 7 language tokens after BOS. The
 * language used is left in w->lang. Writes up to `cap` token ids, returns the
 * count or a negative error. */
int  nd_whistle_decode(nd_whistle *w, const float *enc, uint32_t T, int lang,
                       uint32_t *ids, uint32_t cap);

/* Logits for every position of a token sequence fed from scratch
 * (validation against tools/whistle_ref.py). out is [n][vocab]. */
int  nd_whistle_logits(nd_whistle *w, const float *enc, uint32_t T,
                       const uint32_t *toks, uint32_t n, float *out);

/* Token ids -> UTF-8 text. Returns the length written (NUL-terminated). */
int  nd_whistle_detok(const nd_whistle *w, const uint32_t *ids, uint32_t n,
                      char *out, uint32_t cap);

#ifdef __cplusplus
}
#endif
#endif /* ND_WHISTLE_H */
