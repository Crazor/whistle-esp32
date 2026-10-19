#!/usr/bin/env python3
"""Reader and inspector for whistle.cact (the Needle 3 `.cact` v4 container).

Needle 2's blob is v3 (tag 0x05E12A83, 30-word header, read by check_engine.py).
Whistle ships in v4 (tag 0x05E12A84): a 49-field header describing the text
decoder, the same nameless 44-byte directory and the same CQ blobs, followed by
the speech tensors appended after the decoder. Nothing in the blob names them,
so the canon order is spelled out in SCHEMA below. Every entry was verified by
value against checkpoints/whistle.safetensors from the same Hugging Face repo
(fp16 tensors to rounding, CQ tensors to their quantisation error):

    python3 tools/whistle_cact.py model/whistle/whistle.cact \\
        --verify model/whistle/whistle.safetensors

Without --verify it prints every tensor grouped by encoder and decoder layer,
with the flash each group costs and what each --audio-depth would save.
"""
import argparse
import struct
import sys

import numpy as np

TAG = 0x05E12A84
HDR_FMT = "<48If"
HDR_BYTES = struct.calcsize(HDR_FMT)
REC_FMT = "<BBHIIIIQQII"
REC = struct.calcsize(REC_FMT)
FP16, FP32, CQ, RAW = 1, 2, 3, 4

HDR_FIELDS = (
    "tag num_tensors codebook_len kv_window kv_bits vocab out_vocab d_model "
    "num_heads num_kv_heads num_layers qk_head_dim v_head_dim max_seq_len "
    "hada_n mhc_lanes sliding_window global_mask_lo global_mask_hi "
    "qkv_conv_taps engram_slots engram_sub_dim engram_tables engram_conv_taps "
    "engram_dilation engram_seed_heads num_orders").split()

# Per-layer tensor names, in blob order.
DEC_LAYER = (
    "norm_in q_proj k_proj v_proj q_taps k_taps v_taps q_norm k_norm "
    "gate_proj out_proj post_attn_norm attn_gate pre_hada_norm "
    "mlp.d1 mlp.d2 mlp.b2 mlp.d3 mlp.d4 mlp.w1a mlp.w1b mlp.w2a mlp.w2b "
    "mlp.w3a mlp.w3b mlp.cond_v mlp.cond_u").split()
CROSS_LAYER = (
    "cross_norm q_proj k_proj v_proj gate_proj out_proj q_norm k_norm "
    "post_cross_norm cross_gate").split()
_MLP = "d1 d2 b2 d3 d4 w1a w1b w2a w2b w3a w3b cond_v cond_u".split()
ENC_LAYER = (
    ["pre_hada_norm_0", "norm_in", "post_attn_norm", "conv_norm",
     "conv_out_norm", "pre_hada_norm",
     "q_proj", "k_proj", "v_proj", "gate_proj", "out_proj", "q_norm",
     "k_norm", "attn_gate", "pw1", "pw2", "dw"]
    + ["mlp0." + m for m in _MLP] + ["mlp." + m for m in _MLP])
MHC = ("mhc.a_pre mhc.a_post mhc.a_res mhc.b_pre mhc.b_post mhc.b_res "
       "mhc.phi_pre mhc.phi_post mhc.phi_res").split()
ENGRAM = "tables key_proj value_proj taps".split()
STEM = "w dw_1 pw_1 dw_2 pw_2 out".split()

# Audio manifest (tensor `audio.config`, fp32[13]). The first two words are
# not yet decoded; the rest match the checkpoint's training config.
AUDIO_CONFIG = ("unknown0 unknown1 text_vocab enc_layers enc_lanes "
                "conv_kernel stem_channels stem_stages mels rate fft win "
                "hop").split()


def read_header(raw):
    w = struct.unpack_from(HDR_FMT, raw, 0)
    if w[0] != TAG:
        raise ValueError("not a v4 .cact (tag %#x, want %#x)" % (w[0], TAG))
    h = dict(zip(HDR_FIELDS, w))
    h["orders"] = w[27:27 + h["num_orders"]]
    h["num_sites"] = w[31]
    h["sites"] = w[32:32 + w[31]]
    h["rope_theta"] = w[48]
    return h


def read_directory(raw, h):
    cb = np.frombuffer(raw, np.float32, h["codebook_len"], HDR_BYTES)
    off = HDR_BYTES + 4 * h["codebook_len"]
    recs = []
    for i in range(h["num_tensors"]):
        r = struct.unpack_from(REC_FMT, raw, off + i * REC)
        recs.append({"dtype": r[0], "ndim": r[1], "shape": tuple(r[3:3 + r[1]]),
                     "offset": r[7], "nbytes": r[8], "group": r[9],
                     "bits": r[10]})
    return cb, recs


def schema(h, enc_layers=8):
    """Name every tensor position, given the header. Returns a list of
    (group, name) in blob order."""
    L, S = h["num_layers"], h["num_sites"]
    out = [("decoder", "embedding")]
    for l in range(L):
        out += [("decoder.%d" % l, n) for n in DEC_LAYER]
    out += [("decoder", n) for n in MHC]
    out += [("decoder", "hada_perm1"), ("decoder", "hada_perm2")]
    for s in range(S):
        out += [("engram.%d (layer %d)" % (s, h["sites"][s]), n) for n in ENGRAM]
    out += [("decoder", "final_norm"), ("audio", "audio.config")]
    for l in range(L):
        out += [("decoder.%d" % l, "cross." + n) for n in CROSS_LAYER]
    out += [("encoder", "pe_gate")]
    for l in range(enc_layers):
        out += [("encoder.%d" % l, n) for n in ENC_LAYER]
    out += [("encoder", n) for n in MHC]
    out += [("encoder", "final_norm")]
    out += [("stem", n) for n in STEM]
    out += [("frontend", "mel_filterbank"), ("decoder", "tokenizer")]
    return out


def hadamard(n):
    H = np.array([[1.0]], np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def codebook_for(cb, bits):
    return {2: cb[0:4], 3: cb[4:12], 4: cb[12:28]}[bits]


def dequant(raw, rec, cb):
    """CQ blob -> float32 [out, in]: per group, (codebook[idx] * norm) @ H."""
    out, din = rec["shape"]
    g, b = rec["group"], rec["bits"]
    in_pad = (din + g - 1) // g * g
    rb = in_pad * b // 8
    packed = np.frombuffer(raw, np.uint8, out * rb, rec["offset"]).reshape(out, rb)
    norms = np.frombuffer(raw, np.float16, out * in_pad // g,
                          rec["offset"] + out * rb).astype(np.float32)
    chunks = packed.reshape(out, in_pad // 8, b).astype(np.uint64)
    word = np.zeros(chunks.shape[:-1], np.uint64)
    for k in range(b):
        word |= chunks[..., k] << np.uint64(8 * k)
    mask = np.uint64((1 << b) - 1)
    idx = np.stack([(word >> np.uint64(j * b)) & mask for j in range(8)],
                   -1).reshape(out, in_pad)
    rot = (codebook_for(cb, b)[idx].reshape(out, in_pad // g, g)
           * norms.reshape(out, in_pad // g, 1))
    return (rot @ hadamard(g)).reshape(out, in_pad)[:, :din]


class Cact:
    """A whistle.cact blob with tensors addressable by (group, name)."""

    def __init__(self, path):
        self.raw = open(path, "rb").read()
        self.h = read_header(self.raw)
        self.cb, self.recs = read_directory(self.raw, self.h)
        names = schema(self.h)
        if len(names) != len(self.recs):
            raise ValueError("schema names %d tensors, blob has %d"
                             % (len(names), len(self.recs)))
        self.names = names
        self.index = {gn: i for i, gn in enumerate(names)}
        self._cache = {}

    def t(self, group, name):
        return self.tensor(self.index[(group, name)])

    def tensor(self, i):
        if i in self._cache:
            return self._cache[i]
        r = self.recs[i]
        if r["dtype"] == CQ:
            v = dequant(self.raw, r, self.cb)
        elif r["dtype"] == FP16:
            v = np.frombuffer(self.raw, np.float16, r["nbytes"] // 2,
                              r["offset"]).astype(np.float32).reshape(r["shape"])
        elif r["dtype"] == FP32:
            v = np.frombuffer(self.raw, np.float32, r["nbytes"] // 4,
                              r["offset"]).reshape(r["shape"]).copy()
        else:
            v = self.raw[r["offset"]:r["offset"] + r["nbytes"]]
        self._cache[i] = v
        return v


def fmt(r):
    return {FP16: "fp16", FP32: "fp32", RAW: "raw"}.get(r["dtype"]) or "CQ%d" % r["bits"]


def ladder_order(n):
    """Needle 3's nested decoder-depth order (architecture._ladder_layer_order)."""
    sel, order = [0, n - 1], [0, n - 1]
    while len(order) < n:
        sel.sort()
        _, left, right = max(((r - l, l, r) for l, r in zip(sel, sel[1:]) if r - l > 1),
                             key=lambda t: (t[0], -t[1]))
        sel.append((left + right) // 2)
        order.append((left + right) // 2)
    return order


def inspect(c, verbose):
    h = c.h
    total = len(c.raw)
    print("whistle.cact  %d bytes  %d tensors  tag %#x" % (total, len(c.recs), h["tag"]))
    print("decoder: d_model=%d heads=%d kv_heads=%d layers=%d qk_dim=%d v_dim=%d "
          "vocab=%d qkv_taps=%d engram sites=%s rope=%g" % (
              h["d_model"], h["num_heads"], h["num_kv_heads"], h["num_layers"],
              h["qk_head_dim"], h["v_head_dim"], h["vocab"], h["qkv_conv_taps"],
              list(h["sites"]), h["rope_theta"]))
    ac = c.t("audio", "audio.config")
    print("audio:   " + " ".join("%s=%d" % kv for kv in zip(AUDIO_CONFIG, ac.astype(int))))

    groups = {}
    for i, (g, n) in enumerate(c.names):
        groups.setdefault(g, []).append(i)
    header_dir = HDR_BYTES + 4 * h["codebook_len"] + REC * len(c.recs)

    print()
    for g, idx in groups.items():
        size = sum(c.recs[i]["nbytes"] for i in idx)
        print("%-24s %4d tensors %10s bytes" % (g, len(idx), "{:,}".format(size)))
        if verbose or not g.split(".")[-1].isdigit() or g.endswith(".0"):
            for i in idx:
                r = c.recs[i]
                print("    %4d  %-22s %-14s %-5s %10s" % (
                    i, c.names[i][1], "x".join(map(str, r["shape"])) or "-",
                    fmt(r), "{:,}".format(r["nbytes"])))
    print("%-24s %10s bytes (header, codebook, directory)" % ("", "{:,}".format(header_dir)))

    # Totals and the --audio-depth ladder.
    def sz(pred):
        return sum(r["nbytes"] for (g, _), r in zip(c.names, c.recs) if pred(g))
    L = h["num_layers"]
    per_dec = [sz(lambda g, l=l: g == "decoder.%d" % l) for l in range(L)]
    sites = list(h["sites"])
    per_site = [sz(lambda g, s=s: g.startswith("engram.%d " % s)) for s in range(len(sites))]
    enc = sz(lambda g: g.startswith("encoder") or g in ("stem", "frontend"))
    mhc_row = sum(c.recs[c.index[("decoder", n)]]["nbytes"] for n in MHC) // L
    emb = c.recs[c.index[("decoder", "embedding")]]["nbytes"]
    print()
    print("flash by part:")
    print("    front end + stem + encoder   %10s" % "{:,}".format(enc))
    print("    embedding (CQ4, tied head)   %10s" % "{:,}".format(emb))
    print("    decoder layers (incl. cross) %10s  (%s each)" % (
        "{:,}".format(sum(per_dec)), "{:,}".format(per_dec[0])))
    print("    engram sites                 %10s  (%s each, at layers %s)" % (
        "{:,}".format(sum(per_site)), "{:,}".format(per_site[0]), sites))
    print()
    order = ladder_order(L)
    print("--audio-depth (decoder layers kept, nested; encoder never sliced):")
    print("    depth  layers                   file size     saves")
    for d in range(2, L + 1):
        keep = sorted(order[:d])
        drop = [l for l in range(L) if l not in keep]
        saved = sum(per_dec[l] for l in drop) + sum(
            per_site[s] for s, l in enumerate(sites) if l in drop)
        saved += mhc_row * len(drop)      # sliced mHC rows (approximate: CQ
                                          # norms round per tensor, not row)
        print("    %5d  %-24s %10s  %10s  (%.1f MB)" % (
            d, ",".join(map(str, keep)), "{:,}".format(total - saved),
            "{:,}".format(saved), saved / 1e6))


def verify(c, ckpt_path):
    """Check every named tensor against the training checkpoint, by value."""
    from safetensors.numpy import load_file
    ck = {k: v.astype(np.float32) for k, v in load_file(ckpt_path).items()}

    def src(g, n):
        """Checkpoint slice expected for blob tensor (g, n), in blob layout."""
        if g == "stem":
            v = ck["stem/" + n + ("/kernel" if n.startswith("pw") or n == "out" else "")]
            return v.reshape(9, -1) if v.ndim == 4 else v.T
        if g.startswith("engram."):
            s = int(g.split(".")[1].split()[0])
            v = ck["engrams_%d/%s" % (s, {"tables": "embedding"}.get(n, n) +
                                      ("/kernel" if n.endswith("proj") else ""))]
            return v.reshape(-1, v.shape[-1]) if n == "tables" else (v.T if n.endswith("proj") else v)
        part, layer = ("stack", None) if g.startswith("decoder") else ("encoder", None)
        if g in ("decoder", "encoder"):
            if n == "embedding":
                return ck["embedding/embedding"]
            if n == "pe_gate":
                return ck["pe_gate"].reshape(1)
            if n == "final_norm":
                return ck[part + "/final_norm/scale"]
            if n.startswith("mhc."):
                v = ck[part + "/mhc_" + n[4:]]
                return v.transpose(0, 2, 1).reshape(-1, v.shape[1]) if "phi" in n else v
            return None
        layer = int(g.split(".")[1])
        blk = part + "/layers/block/"
        if n.startswith("cross."):
            n, attn = n[6:], "cross_attn/"
        else:
            attn = "self_attn/"
        mlp = {"mlp0.": "hadamard_mlp_0/", "mlp.": "hadamard_mlp/"}
        for p, k in mlp.items():
            if n.startswith(p):
                return ck[blk + k + n[len(p):]][layer]
        if n in ("q_proj", "k_proj", "v_proj", "gate_proj", "out_proj"):
            return ck[blk + attn + n + "/kernel"][layer].T
        if n in ("q_taps", "k_taps", "v_taps"):
            return ck[blk + attn + n][layer]
        if n in ("q_norm", "k_norm"):
            return ck[blk + attn + n + "/scale"][layer]
        if n in ("attn_gate", "cross_gate"):
            return ck[blk + n][layer].reshape(1)
        if n in ("pw1", "pw2"):
            return ck[blk + n + "/kernel"][layer].T
        if n == "dw":
            return ck[blk + "dw"][layer].reshape(9, -1)
        if n == "norm_in":
            return ck[blk + "ZCRMSNorm_0/scale"][layer]
        return ck[blk + n + "/scale"][layer]

    worst = {"fp16": 0.0, "fp32": 0.0, "CQ2": 0.0, "CQ4": 0.0}
    seen, bad = 0, 0
    for i, (g, n) in enumerate(c.names):
        r = c.recs[i]
        if n in ("tokenizer", "mel_filterbank", "audio.config", "hada_perm1", "hada_perm2"):
            continue
        ref = src(g, n)
        a = c.tensor(i)
        if ref is None or ref.shape != a.shape:
            print("MISMATCH %d %s/%s blob %s ckpt %s" % (
                i, g, n, a.shape, None if ref is None else ref.shape))
            bad += 1
            continue
        err = float(np.linalg.norm(a - ref) / (np.linalg.norm(ref) + 1e-12))
        k = fmt(r)
        worst[k] = max(worst[k], err)
        # fp16 rounding is ~2e-4; CQ2 / CQ4 quantisation error ~0.36 / ~0.11.
        limit = {"fp16": 1e-2, "fp32": 1e-6, "CQ2": 0.45, "CQ4": 0.15}[k]
        if err > limit:
            print("BAD %d %s/%s %s rel err %.3g" % (i, g, n, k, err))
            bad += 1
        seen += 1

    # The two permutations and the filterbank are generated, not trained.
    from numpy.random import RandomState
    for name, seed in (("hada_perm1", 11), ("hada_perm2", 13)):
        p = c.t("decoder", name).astype(int)
        ok = np.array_equal(p, RandomState(seed).permutation(len(p)))
        print("%s == RandomState(%d).permutation(%d): %s" % (name, seed, len(p), ok))
        bad += not ok
    print("verified %d tensors by value; worst rel err: %s; problems: %d" % (
        seen, ", ".join("%s %.3g" % kv for kv in worst.items()), bad))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("--verify", metavar="SAFETENSORS")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="list every layer's tensors, not just layer 0's")
    a = ap.parse_args()
    c = Cact(a.blob)
    if a.verify:
        return 1 if verify(c, a.verify) else 0
    inspect(c, a.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
