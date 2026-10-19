#!/usr/bin/env python3
"""Write a shallower whistle.cact: only the decoder layers a ladder depth keeps.

  python3 tools/whistle_strip.py model/whistle/whistle.cact 4 model/whistle/whistle_d4.cact

Whistle's decoder is laddered (needle/model/architecture.py): depth k is the
nested subset ladder_order[:k], trained as a model of its own, and the
official engine selects it at load time with --audio-depth. A board has no
room to carry the layers it will never run, so this rewrites the blob the way
architecture.ladder_slice slices the parameters:

  - the kept layers' 27 decoder tensors and 10 cross-attention tensors
  - their rows of the 9 mHC tensors (the CQ phi tensors are row-sliced:
    packed rows, then the matching group norms)
  - the engram sites whose layer is kept, renumbered
  - the header's num_layers, num_sites and sites[] updated to match

Everything else (encoder, stem, filterbank, embedding, tokenizer) is copied
byte for byte. The engine needs no depth setting for the result: its lane
offsets come from a layer's position in the stack, which is what the
ladder's sub-lane offsets are (architecture._ScanBody, sub_pre_off).
"""
import argparse
import struct
import sys

import numpy as np

from whistle_cact import (CQ, HDR_BYTES, HDR_FMT, MHC, REC_FMT, Cact,
                          ladder_order)

ALIGN = 64


def rec_bytes(r):
    return struct.pack(REC_FMT, r["dtype"], len(r["shape"]), 0,
                       *(list(r["shape"]) + [0] * (4 - len(r["shape"]))),
                       0, len(r["blob"]), r["group"], r["bits"])


def slice_rows(c, i, rows):
    """Tensor i restricted to `rows` (leading axis), as (shape, blob)."""
    r = c.recs[i]
    raw = c.raw[r["offset"]:r["offset"] + r["nbytes"]]
    if r["dtype"] == CQ:
        out, din = r["shape"]
        g, b = r["group"], r["bits"]
        in_pad = (din + g - 1) // g * g
        rb, nb = in_pad * b // 8, in_pad // g * 2
        packed = np.frombuffer(raw, np.uint8, out * rb).reshape(out, rb)[rows]
        norms = np.frombuffer(raw, np.uint8, out * nb, out * rb).reshape(out, nb)[rows]
        return (len(rows), din), packed.tobytes() + norms.tobytes()
    item = 2 if r["dtype"] == 1 else 4
    shape = r["shape"]
    per = int(np.prod(shape[1:])) * item if len(shape) > 1 else item
    blob = b"".join(raw[k * per:(k + 1) * per] for k in rows)
    return (len(rows),) + tuple(shape[1:]), blob


def strip(src, depth, dst):
    c = Cact(src)
    h = c.h
    L, n = h["num_layers"], h["mhc_lanes"]
    if not 2 <= depth <= L:
        raise SystemExit("depth must be 2..%d" % L)
    keep = sorted(ladder_order(L)[:depth])
    sites = [(s, l) for s, l in enumerate(h["sites"]) if l in keep]

    out = []                                    # list of record dicts with "blob"

    def copy(i):
        r = dict(c.recs[i])
        r["blob"] = c.raw[r["offset"]:r["offset"] + r["nbytes"]]
        out.append(r)

    def copy_group(g):
        for i, (gg, _) in enumerate(c.names):
            if gg == g:
                copy(i)

    copy(c.index[("decoder", "embedding")])
    for l in keep:
        for i, (gg, nm) in enumerate(c.names):
            if gg == "decoder.%d" % l and not nm.startswith("cross."):
                copy(i)
    for nm in MHC:
        i = c.index[("decoder", nm)]
        if "phi" in nm:
            per = c.recs[i]["shape"][0] // L     # rows per layer: n or n*n
            rows = [l * per + k for l in keep for k in range(per)]
        else:
            rows = keep
        shape, blob = slice_rows(c, i, rows)
        r = dict(c.recs[i])
        r.update(shape=shape, blob=blob)
        out.append(r)
    copy(c.index[("decoder", "hada_perm1")])
    copy(c.index[("decoder", "hada_perm2")])
    for s, l in sites:
        copy_group("engram.%d (layer %d)" % (s, l))
    copy(c.index[("decoder", "final_norm")])
    copy(c.index[("audio", "audio.config")])
    for l in keep:
        for i, (gg, nm) in enumerate(c.names):
            if gg == "decoder.%d" % l and nm.startswith("cross."):
                copy(i)
    first_enc = c.index[("encoder", "pe_gate")]
    for i in range(first_enc, len(c.recs)):
        copy(i)

    # Header: same geometry, fewer layers and sites.
    w = list(struct.unpack_from(HDR_FMT, c.raw, 0))
    w[1] = len(out)
    w[10] = depth
    w[31] = len(sites)
    for k in range(16):
        w[32 + k] = keep.index(sites[k][1]) if k < len(sites) else 0
    hdr = struct.pack(HDR_FMT, *w)
    cb = c.raw[HDR_BYTES:HDR_BYTES + 4 * h["codebook_len"]]

    off = HDR_BYTES + len(cb) + len(out) * struct.calcsize(REC_FMT)
    off = (off + ALIGN - 1) // ALIGN * ALIGN
    body = bytearray()
    for r in out:
        r["new_offset"] = off + len(body)
        body += r["blob"]
        body += bytes((-len(body)) % ALIGN)
    dir_bytes = b""
    for r in out:
        b = bytearray(rec_bytes(r))
        struct.pack_into("<Q", b, 20, r["new_offset"])
        dir_bytes += bytes(b)
    head = hdr + cb + dir_bytes
    head += bytes(off - len(head))
    data = head + bytes(body)
    data = data[:off + len(body) - ((-len(out[-1]["blob"])) % ALIGN)]   # no trailing pad
    open(dst, "wb").write(data)
    return keep, [l for _, l in sites], len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("depth", type=int)
    ap.add_argument("out")
    a = ap.parse_args()
    keep, sites, size = strip(a.blob, a.depth, a.out)
    print("depth %d: decoder layers %s, engram at %s -> %s (%d bytes, %.2f MB)"
          % (a.depth, keep, sites, a.out, size, size / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
