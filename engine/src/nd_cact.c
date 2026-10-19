#include "nd_cact.h"

#include <string.h>

/* The header is a flat run of 30 little-endian u32-sized fields and
 * nd_cact_header mirrors that run exactly, so one bounds-checked copy decodes
 * it. (Every field is 4 bytes, so the struct has no interior padding.) */
#define ND_HDR_WORDS 30
#define ND_HDR_BYTES (ND_HDR_WORDS * 4)
#define ND_HDR4_WORDS 49    /* v4: same idea, 49 four-byte fields */
#define ND_HDR4_BYTES (ND_HDR4_WORDS * 4)

static uint32_t rd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t rd_u64(const uint8_t *p)
{
    return (uint64_t)rd_u32(p) | ((uint64_t)rd_u32(p + 4) << 32);
}

/* Codebook and directory follow the header in both versions. */
static int open_tail(nd_cact *c, size_t hdr_bytes, uint32_t num_tensors,
                     uint32_t codebook_len)
{
    size_t cb_off  = hdr_bytes;
    size_t cb_len  = (size_t)codebook_len * 4;
    size_t dir_off = cb_off + cb_len;
    size_t dir_len = (size_t)num_tensors * ND_CACT_REC_SIZE;

    if (cb_off + cb_len < cb_off || dir_off + dir_len < dir_off ||
        dir_off + dir_len > c->size)
        return -4;

    /* cb2[4] | cb3[8] | cb4[16] */
    if (codebook_len < 28)
        return -5;

    c->codebook  = (const float *)(const void *)(c->base + cb_off);
    c->directory = c->base + dir_off;
    c->n         = num_tensors;
    return 0;
}

static int open_v4(nd_cact *c)
{
    uint32_t words[ND_HDR4_WORDS];
    uint32_t i;

    if (c->size < ND_HDR4_BYTES)
        return -1;
    for (i = 0; i < ND_HDR4_WORDS; i++)
        words[i] = rd_u32(c->base + i * 4);
    /* nd_cact_header4 is 49 four-byte fields with no interior padding. */
    memcpy(&c->h4, words, sizeof(c->h4));
    c->version = 4;

    if (c->h4.d_model == 0 || c->h4.num_layers == 0 || c->h4.num_heads == 0 ||
        c->h4.num_kv_heads == 0 || c->h4.qk_head_dim == 0 ||
        c->h4.v_head_dim == 0 || c->h4.num_heads % c->h4.num_kv_heads != 0 ||
        c->h4.num_orders > 4 || c->h4.num_sites > 16)
        return -3;
    return open_tail(c, ND_HDR4_BYTES, c->h4.num_tensors, c->h4.codebook_len);
}

int nd_cact_open(nd_cact *c, const void *blob, size_t size)
{
    const uint8_t *b = (const uint8_t *)blob;

    if (!c || !b || size < ND_HDR_BYTES)
        return -1;

    memset(c, 0, sizeof(*c));
    c->base = b;
    c->size = size;

    if (rd_u32(b) == ND_CACT_TAG4)
        return open_v4(c);

    /* Decode the header word by word rather than casting: the blob is only
     * 64-byte aligned at tensor boundaries and may be an mmap'd flash window,
     * and this keeps the reader endian-explicit. */
    {
        uint32_t words[ND_HDR_WORDS];
        uint32_t i;
        for (i = 0; i < ND_HDR_WORDS; i++)
            words[i] = rd_u32(b + i * 4);
        memcpy(&c->h, words, sizeof(c->h));
    }

    if (c->h.tag != ND_CACT_TAG)
        return -2;
    c->version = 3;

    /* Only the geometry this engine actually implements. */
    if (c->h.d_model == 0 || c->h.num_layers == 0 || c->h.num_heads == 0 ||
        c->h.num_kv_heads == 0 || c->h.head_dim == 0 ||
        c->h.num_heads % c->h.num_kv_heads != 0 ||
        c->h.num_heads * c->h.head_dim != c->h.attn_dim ||
        c->h.num_orders > 4 || c->h.num_sites > 4)
        return -3;

    return open_tail(c, ND_HDR_BYTES, c->h.num_tensors, c->h.codebook_len);
}

const float *nd_cact_codebook(const nd_cact *c, uint32_t bits)
{
    switch (bits) {
    case 2: return c->codebook;          /* 4 entries  */
    case 3: return c->codebook + 4;      /* 8 entries  */
    case 4: return c->codebook + 12;     /* 16 entries */
    default: return NULL;
    }
}

int nd_cact_tensor(const nd_cact *c, uint32_t i, nd_tensor *t)
{
    const uint8_t *r;

    if (!c || !t || i >= c->n)
        return -1;

    r = c->directory + (size_t)i * ND_CACT_REC_SIZE;

    t->dtype    = r[0];
    t->ndim     = r[1];
    /* r[2..3] is padding */
    t->shape[0] = rd_u32(r + 4);
    t->shape[1] = rd_u32(r + 8);
    t->shape[2] = rd_u32(r + 12);
    t->shape[3] = rd_u32(r + 16);
    t->offset   = rd_u64(r + 20);
    t->nbytes   = rd_u64(r + 28);
    t->group    = rd_u32(r + 36);
    t->bits     = rd_u32(r + 40);

    /* Shape words past ndim are written as zero; normalise them so callers
     * can treat shape[] uniformly. */
    {
        uint32_t d;
        for (d = t->ndim; d < 4; d++)
            t->shape[d] = 0;
    }
    return 0;
}

const void *nd_cact_data(const nd_cact *c, const nd_tensor *t)
{
    if (t->offset > c->size || t->nbytes > c->size - t->offset)
        return NULL;
    return c->base + t->offset;
}
