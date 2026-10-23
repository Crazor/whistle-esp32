/* nd_comp.c - see nd_comp.h. */
#include "nd_comp.h"

#include <math.h>

static float db_to_lin(float db) { return powf(10.0f, db / 20.0f); }

void nd_comp_init(nd_comp *c, int rate, float ceiling_db)
{
    c->env = 0.0f;
    c->att = 1.0f - expf(-1.0f / (0.001f * (float)rate));
    c->rel = 1.0f - expf(-1.0f / (0.120f * (float)rate));
    c->thresh = db_to_lin(ND_COMP_THRESH_DB);
    c->slope = 1.0f / ND_COMP_RATIO - 1.0f;
    c->makeup = db_to_lin(ND_COMP_MAKEUP_DB);
    c->ceiling = db_to_lin(ceiling_db > 0.0f ? 0.0f : ceiling_db);
}

void nd_comp_run(nd_comp *c, int16_t *pcm, size_t n)
{
    const float knee = 0.9f * c->ceiling, room = c->ceiling - knee;
    size_t      i;

    for (i = 0; i < n; i++) {
        float x = pcm[i] * (1.0f / 32768.0f);
        float a = fabsf(x), g, y, m;

        c->env += (a > c->env ? c->att : c->rel) * (a - c->env);
        /* above the threshold, level^(1/ratio): gain (env/thresh)^(1/r - 1) */
        g = c->env > c->thresh ? powf(c->env / c->thresh, c->slope) : 1.0f;
        y = x * g * c->makeup;

        m = fabsf(y);
        if (m > knee)                   /* soft knee, never past ceiling */
            y = copysignf(knee + room * tanhf((m - knee) / room), y);
        pcm[i] = (int16_t)lrintf(y * 32767.0f);
    }
}
