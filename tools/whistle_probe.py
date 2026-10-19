#!/usr/bin/env python3
"""Probe blobs: copies of whistle.cact with chosen tensors overwritten, run
through the official engine (pip install cactus-needle) so its intermediate
behaviour can be observed through embed(). Used to pin down the parts of the
architecture with no published source; see whistle_ref.py.
"""
import os
import struct
import tempfile

import numpy as np

from whistle_cact import CQ, FP16, FP32, Cact


def write_probe(src, edits, path):
    """edits: {(group, name): array or 0}. 0 zeroes a tensor (for CQ, its
    group norms); an array overwrites an fp16/fp32 tensor in place.

    Use a fresh path for every distinct probe. needle.Whistle skips reloading
    when the weights path matches the one already loaded (agent/whistle.py
    _bind), so a rewritten file at the same path is never re-read: every
    result after the first silently comes from the old weights."""
    if os.path.exists(path):
        raise FileExistsError("probe path reused: %s" % path)
    c = Cact(src)
    raw = bytearray(c.raw)
    for key, val in edits.items():
        r = c.recs[c.index[key]]
        if r["dtype"] == CQ:
            assert isinstance(val, int) and val == 0, "CQ tensors can only be zeroed"
            out, din = r["shape"]
            g, b = r["group"], r["bits"]
            in_pad = (din + g - 1) // g * g
            norms_off = r["offset"] + out * in_pad * b // 8
            raw[norms_off:r["offset"] + r["nbytes"]] = bytes(r["offset"] + r["nbytes"] - norms_off)
        else:
            dt = np.float16 if r["dtype"] == FP16 else np.float32
            a = np.zeros(r["shape"], dt) if isinstance(val, int) else np.asarray(val, dt).reshape(r["shape"])
            raw[r["offset"]:r["offset"] + r["nbytes"]] = a.tobytes()
    open(path, "wb").write(raw)
    return path


def engine_embed(blob, audio):
    """Encoder output [frames, d] from the official engine on `blob`."""
    os.environ["NEEDLE_TELEMETRY"] = "0"
    import needle
    w = needle.Whistle(weights=blob)
    return np.array(w.embed(np.asarray(audio, np.float32).tolist())).reshape(-1, 512)


def identity_encoder(c):
    """Edits that make every encoder block return its input unchanged."""
    e = {}
    for l in range(8):
        g = "encoder.%d" % l
        e[(g, "out_proj")] = 0
        e[(g, "pw2")] = 0
        e[(g, "mlp0.d4")] = 0
        e[(g, "mlp.d4")] = 0
    return e
