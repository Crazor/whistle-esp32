#!/usr/bin/env python3
"""NumPy reference forward pass for Whistle, read straight from whistle.cact.

The counterpart of ref_forward.py (Needle 2). Whistle's speech architecture
has no published source: the shipped engine is a prebuilt library. The
decoder and the encoder's attention / Hadamard MLP / mHC are Needle 3 blocks,
transcribed here from needle/model/architecture.py. The front end, stem,
encoder block layout and cross attention were established by probing the
official engine (tools/check_whistle.py), and the choices that probing pinned
down are marked PROBED below.

  python3 tools/whistle_ref.py model/whistle/whistle.cact clip.wav
"""
import argparse
import sys
import wave

import numpy as np

from whistle_cact import Cact

EPS = 1e-6


# ------------------------------------------------------------------ helpers

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def silu(x):
    return x * sigmoid(x)


def rms_unit(x):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + EPS)


def zcrms(x, scale):
    return (1.0 + scale) * rms_unit(x)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def sinkhorn(logits, iters=20):
    k = logits.astype(np.float64)
    for _ in range(iters):
        k = k - _lse(k, -1)
        k = k - _lse(k, -2)
    return np.exp(k).astype(np.float32)


def _lse(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))


def rope(z, theta):
    """Half-split rotary over the last axis; z is [heads, T, hd]."""
    T, hd = z.shape[-2], z.shape[-1]
    half = hd // 2
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
    ang = np.arange(T, dtype=np.float32)[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    z1, z2 = z[..., :half], z[..., half:]
    return np.concatenate([z1 * cos - z2 * sin, z2 * cos + z1 * sin], -1)


def load_wav(path):
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, \
        "want 16 kHz mono s16"
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


# ------------------------------------------------------------------ model

class Whistle:
    def __init__(self, path, **opts):
        self.c = Cact(path)
        self.h = self.c.h
        self.d = self.h["d_model"]
        ac = self.c.t("audio", "audio.config").astype(int)
        self.win, self.hop, self.nfft = int(ac[11]), int(ac[12]), int(ac[10])
        self.opts = dict(DEFAULTS)
        self.opts.update(opts)

    def mm(self, x, w):
        """x @ w.T, with the engine's int8 activations (CQ-WxA8) when act_q8:
        symmetric, one scale per row, applied to the matmul input."""
        if self.opts["act_q8"]:
            s = np.abs(x).max(-1, keepdims=True) / 127.0
            s = np.where(s > 0, s, 1.0)
            x = np.clip(np.rint(x / s), -127, 127) * s
        return x @ w.T

    # -------------------------------------------------------------- front end
    def logmel(self, x):
        o = self.opts
        n_frames = len(x) // self.hop                    # PROBED: n // hop frames
        if o["gain"] == "peak":
            x = x / max(np.abs(x).max(), 1e-9)
        elif o["gain"] == "rms":
            x = x / max(np.sqrt((x ** 2).mean()), 1e-9)
        if o["center"]:
            pad = self.nfft // 2 if o["center_pad"] == "nfft" else self.win // 2
            xp = np.pad(x, (pad, pad), mode=o["pad_mode"])
        else:
            xp = np.pad(x, (0, self.win))
        win = np.hanning(self.win + 1)[:-1] if o["periodic"] else np.hanning(self.win)
        win = win.astype(np.float32)
        span = self.nfft if (o["center"] and o["center_pad"] == "nfft" and o["frame_len"] == "nfft") else self.win
        idx = np.arange(n_frames)[:, None] * self.hop + np.arange(span)[None, :]
        fr = xp[idx]
        if span == self.nfft:
            w = np.zeros(self.nfft, np.float32)
            off = (self.nfft - self.win) // 2
            w[off:off + self.win] = win
            fr = fr * w
        else:
            fr = fr * win
        spec = np.abs(np.fft.rfft(fr, self.nfft, axis=-1)) ** o["power"]
        mel = spec @ self.c.t("frontend", "mel_filterbank")
        if o["log"] == "rel":                             # floor relative to the loudest bin
            lm = np.log10(np.maximum(mel, 1e-10))
            lm = np.maximum(lm, lm.max() - o["log_floor"])
        elif o["log"] == "ln":
            lm = np.log(np.maximum(mel, o["log_floor"]))
        elif o["log"] == "ln_add":
            lm = np.log(mel + o["log_floor"])
        else:
            lm = np.log10(np.maximum(mel, o["log_floor"]))
        if o["norm"] == "channel":                        # per mel channel over time
            lm = (lm - lm.mean(0, keepdims=True)) / (lm.std(0, keepdims=True, ddof=o["ddof"]) + o["norm_eps"])
        elif o["norm"] == "global":
            lm = (lm - lm.mean()) / (lm.std() + o["norm_eps"])
        return lm.astype(np.float32)                      # [frames, mels]

    # -------------------------------------------------------------- stem
    def _conv3(self, x, w, stride, depthwise):
        """3x3 'same' conv on [T, F, C] with stride on both axes.
        w is [9, C] (depthwise or single input channel), index kt*3+kf."""
        T, F, C = x.shape
        xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
        To, Fo = (T + stride - 1) // stride, (F + stride - 1) // stride
        k = w.reshape(3, 3, -1)
        if self.opts["stem_kernel_tf"] == "ft":
            k = k.transpose(1, 0, 2)
        out = np.zeros((To, Fo, k.shape[-1]), np.float32)
        for a in range(3):
            for b in range(3):
                patch = xp[a:a + stride * To:stride, b:b + stride * Fo:stride, :]
                out += patch * k[a, b] if depthwise else patch[..., :1] * k[a, b]
        return out

    def stem(self, lm):
        c, o = self.c, self.opts
        act = {"silu": silu, "gelu": lambda z: 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3))),
               "relu": lambda z: np.maximum(z, 0)}[o["stem_act"]]
        x = lm[:, :, None]
        x = act(self._conv3(x, c.t("stem", "w"), 2, False))
        x = self._conv3(x, c.t("stem", "dw_1"), 2, True)
        x = self.mm(x.reshape(-1, x.shape[-1]), c.t("stem", "pw_1")).reshape(x.shape[0], x.shape[1], -1)
        x = act(x)
        x = self._conv3(x, c.t("stem", "dw_2"), 2, True)
        x = self.mm(x.reshape(-1, x.shape[-1]), c.t("stem", "pw_2")).reshape(x.shape[0], x.shape[1], -1)
        x = act(x)
        T, F, C = x.shape
        flat = x.reshape(T, F * C) if o["stem_flatten"] == "fc" else x.transpose(0, 2, 1).reshape(T, C * F)
        y = self.mm(flat, c.t("stem", "out"))
        return y * (np.sqrt(np.float32(self.d)) if o["stem_scale"] else 1.0)

    def posenc(self, T):
        d = self.d
        pos = np.arange(T, dtype=np.float32)[:, None]
        div = np.exp(np.arange(0, d, 2, dtype=np.float32) * (-np.log(10000.0) / d))
        pe = np.zeros((T, d), np.float32)
        pe[:, 0::2], pe[:, 1::2] = np.sin(pos * div), np.cos(pos * div)
        return pe

    # -------------------------------------------------------------- shared blocks
    def hada_mlp(self, x, get, prefix):
        """Needle 3 HadamardMLP (Monarch: Kronecker 16x32 factors + perms)."""
        n = x.shape[-1]
        p1 = self.c.t("decoder", "hada_perm1").astype(int)
        p2 = self.c.t("decoder", "hada_perm2").astype(int)

        def kron(z, a, b):
            z = z.reshape(-1, a.shape[0], b.shape[0])
            return np.einsum("tij,ik,jl->tkl", z, a, b).reshape(-1, n)

        g = lambda k: get(prefix + k)
        cond = 1.0 + softmax(x @ g("cond_v")) @ g("cond_u")
        z = kron(g("d1") * x, g("w1a"), g("w1b"))[:, p1]
        z = kron(silu(g("d2") * cond * z + g("b2")), g("w2a"), g("w2b"))[:, p2]
        z = kron(g("d3") * z, g("w3a"), g("w3b"))
        return g("d4") * z

    def attention(self, xq, xkv, get, nh, nkv, causal, taps, use_rope):
        """Needle 3 MultiHeadAttention; xkv != xq gives cross attention."""
        h = self.h
        qk, vd = h["qk_head_dim"], h["v_head_dim"]
        Tq, Tk = xq.shape[0], xkv.shape[0]
        q = self.mm(xq, get("q_proj"))
        k = self.mm(xkv, get("k_proj"))
        v = self.mm(xkv, get("v_proj"))
        if taps:
            def tap(z, t):
                out = np.zeros_like(z)
                for j in range(t.shape[0]):
                    out[j:] += t[j] * z[:z.shape[0] - j]
                return out
            q, k, v = tap(q, get("q_taps")), tap(k, get("k_taps")), tap(v, get("v_taps"))
        q = q.reshape(Tq, nh, qk).transpose(1, 0, 2)
        k = k.reshape(Tk, nkv, qk).transpose(1, 0, 2)
        v = v.reshape(Tk, nkv, vd).transpose(1, 0, 2)
        q = zcrms(q, get("q_norm"))
        k = zcrms(k, get("k_norm"))
        if use_rope:
            q, k = rope(q, h["rope_theta"]), rope(k, h["rope_theta"])
        if self.opts["kv_q8"]:
            # int8 K/V, one scale per (head, position), after rope - as the
            # header's kv_bits=8 and Needle 2's cache (ref_forward --int8-kv).
            def q8(z):
                sc = np.abs(z).max(-1, keepdims=True) / 127.0
                sc = np.where(sc > 0, sc, 1.0)
                return np.clip(np.rint(z / sc), -127, 127) * sc
            k, v = q8(k), q8(v)
            if self.opts["q_q8"]:
                q = q8(q)
        rep = nh // nkv
        k, v = np.repeat(k, rep, 0), np.repeat(v, rep, 0)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(np.float32(qk))
        if causal:
            s = np.where(np.tril(np.ones((Tq, Tk), bool))[None], s, -np.inf)
        out = (softmax(s) @ v).transpose(1, 0, 2).reshape(Tq, nh * vd)
        out = out * sigmoid(self.mm(xq, get("gate_proj")))
        return self.mm(out, get("out_proj"))

    def mhc_stack(self, x, part, L, block):
        """Needle 3 Stack: n residual lanes mixed by mHC around each block."""
        c, d = self.c, self.d
        n = self.h["mhc_lanes"]
        T = x.shape[0]
        g = lambda k: c.t(part, "mhc." + k)
        phi_pre = g("phi_pre").reshape(L, n, n * d).transpose(0, 2, 1)
        phi_post = g("phi_post").reshape(L, n, n * d).transpose(0, 2, 1)
        phi_res = g("phi_res").reshape(L, n * n, n * d).transpose(0, 2, 1)
        lane = np.eye(n, dtype=np.float32)[np.arange(L) % n]
        pre_off, post_off = 8 * lane - 4, -4 * (1 - lane)
        xl = np.broadcast_to(x[:, None, :], (T, n, d)).astype(np.float32).copy()
        for i in range(L):
            nx = rms_unit(xl.reshape(T, n * d))
            nxq = nx if not self.opts["act_q8"] else self.mm(nx, np.eye(n * d, dtype=np.float32))
            hpre = sigmoid(g("a_pre")[i] * (nxq @ phi_pre[i]) + g("b_pre")[i] + pre_off[i])
            u = np.einsum("tn,tnc->tc", hpre, xl)
            y = block(i, u) - u
            hpost = 2 * sigmoid(g("a_post")[i] * (nxq @ phi_post[i]) + g("b_post")[i] + post_off[i])
            hres = sinkhorn(g("a_res")[i] * (nxq @ phi_res[i]).reshape(T, n, n) + g("b_res")[i])
            xl = np.einsum("tij,tjc->tic", hres, xl) + hpost[:, :, None] * y[:, None, :]
        return xl.mean(1)

    # -------------------------------------------------------------- encoder
    def enc_block(self, i, x):
        c, o, h = self.c, self.opts, self.h
        get = lambda k: c.t("encoder.%d" % i, k)
        half = 0.5 if o["macaron_half"] else 1.0

        x = x + half * self.hada_mlp(zcrms(x, get("pre_hada_norm_0")), get, "mlp0.")

        xn = zcrms(x, get("norm_in"))
        t = self.attention(xn, xn,
                           get, h["num_heads"], h["num_kv_heads"], False, False, o["enc_rope"])
        x = x + sigmoid(get("attn_gate")[0]) * zcrms(t, get("post_attn_norm"))

        t = self.mm(zcrms(x, get("conv_norm")), get("pw1"))
        a, b = t[:, :self.d], t[:, self.d:]
        t = a * sigmoid(b) if o["glu"] == "sigmoid" else a * silu(b)
        dw = get("dw")                                      # [9, d], centred
        K = dw.shape[0]
        tp = np.pad(t, ((K // 2, K // 2), (0, 0)))
        t = sum(dw[j] * tp[j:j + t.shape[0]] for j in range(K))
        if o["conv_out_norm"] == "mid":
            t = self.mm(silu(zcrms(t, get("conv_out_norm"))), get("pw2"))
        else:
            t = zcrms(self.mm(silu(t), get("pw2")), get("conv_out_norm"))
        x = x + t

        x = x + half * self.hada_mlp(zcrms(x, get("pre_hada_norm")), get, "mlp.")
        return x

    def encode(self, audio, identity=False, upto=8):
        """Encoder output [frames, d] - what needle.Whistle().embed returns.
        identity / upto make blocks (all / from `upto` on) pass-through, to
        line up with the probe blobs in whistle_probe.py."""
        x = self.stem(self.logmel(audio))
        if self.opts["pe"]:
            x = x + sigmoid(self.c.t("encoder", "pe_gate")[0]) * self.posenc(x.shape[0])
        blk = (lambda i, u: u if identity or i >= upto else self.enc_block(i, u))
        x = self.mhc_stack(x, "encoder", 8, blk)
        return zcrms(x, self.c.t("encoder", "final_norm"))



# ------------------------------------------------------------------ decoder

EG_SEED, EG_PRIME = 0x9E3779B9, 0x01000193
LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")
PAD, EOS, BOS = 0, 1, 2


class Tokenizer:
    """Decode side of the RAW SentencePiece dump appended to the blob
    (format in needle/model/export.py)."""

    def __init__(self, blob):
        import struct
        n, self.pad, self.eos, self.bos, self.unk, self.dummy, self.byte_fb, _ = \
            struct.unpack_from("<IIIIIBBH", blob, 0)
        off, self.pieces, self.types, self.scores = struct.calcsize("<IIIIIBBH"), [], [], []
        for _ in range(n):
            sc, ty, ln = struct.unpack_from("<fBH", blob, off)
            off += 7
            self.pieces.append(bytes(blob[off:off + ln]).decode("utf-8", "replace"))
            self.types.append(ty)
            self.scores.append(sc)
            off += ln

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            if i >= len(self.pieces) or self.types[i] in (1, 2):
                continue
            p = self.pieces[i]
            if self.types[i] == 4:                      # <0xNN> byte piece
                out.append(int(p[3:5], 16))
            else:
                out += p.replace("\u2581", " ").encode()
        s = out.decode("utf-8", "replace")
        return s[1:] if s.startswith(" ") else s


def _engram_indices(tokens, orders, heads, slots):
    u = np.asarray(tokens, np.uint32)
    T = len(u)
    idx, ok = [], []
    for oi, order in enumerate(orders):
        for hh in range(heads):
            acc = np.full(T, np.uint32((EG_SEED * (oi * heads + hh + 1)) & 0xFFFFFFFF), np.uint32)
            for j in range(order):
                sh = np.zeros(T, np.uint32)
                sh[j:] = u[:T - j]
                acc = (acc ^ sh) * np.uint32(EG_PRIME)
            acc = acc ^ (acc >> np.uint32(15))
            idx.append(acc % np.uint32(slots))
            ok.append((np.arange(T) >= order - 1).astype(np.float32))
    return np.stack(idx, -1).astype(np.int64), np.stack(ok, -1)


class Decoder:
    """Whistle decoder: Needle 3 blocks plus gated cross attention, read
    non-incrementally over the whole token sequence (like ref_forward.py)."""

    def __init__(self, model):
        self.m, self.c, self.h = model, model.c, model.h
        self.tok = Tokenizer(self.c.t("decoder", "tokenizer"))

    def engram_kv(self, tokens):
        c, h = self.c, self.h
        orders = list(h["orders"])
        heads = h["engram_tables"] // len(orders)
        idx, ok = _engram_indices(tokens, orders, heads, h["engram_slots"])
        T, out = len(tokens), []
        for s in range(h["num_sites"]):
            g = lambda n: c.t("engram.%d (layer %d)" % (s, h["sites"][s]), n)
            tables = g("tables").reshape(h["engram_tables"], h["engram_slots"], h["engram_sub_dim"])
            e = (tables[np.arange(h["engram_tables"])[None, :], idx] * ok[:, :, None]).reshape(T, -1)
            k, v = self.m.mm(e, g("key_proj")), self.m.mm(e, g("value_proj"))
            taps, dil = g("taps"), h["engram_dilation"]
            acc = np.zeros_like(v)
            for j in range(taps.shape[0]):
                sh = j * dil
                if sh < T:
                    acc[sh:] += taps[j] * v[:T - sh]
            out.append((k, acc))
        return out

    def cross_kv(self, enc):
        """Per-layer cross-attention K, V from the encoder output - computed
        once per clip, as the blog describes."""
        h = self.h
        nh, qk, vd = h["num_heads"], h["qk_head_dim"], h["v_head_dim"]
        kv = []
        for l in range(h["num_layers"]):
            get = lambda n: self.c.t("decoder.%d" % l, "cross." + n)
            k = self.m.mm(enc, get("k_proj")).reshape(-1, nh, qk).transpose(1, 0, 2)
            v = self.m.mm(enc, get("v_proj")).reshape(-1, nh, vd).transpose(1, 0, 2)
            kv.append((zcrms(k, get("k_norm")), v))
        return kv

    def cross(self, l, x, kv):
        h = self.h
        nh, qk, vd = h["num_heads"], h["qk_head_dim"], h["v_head_dim"]
        get = lambda n: self.c.t("decoder.%d" % l, "cross." + n)
        k, v = kv
        q = zcrms(self.m.mm(x, get("q_proj")).reshape(-1, nh, qk).transpose(1, 0, 2), get("q_norm"))
        a = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(np.float32(qk)))
        out = (a @ v).transpose(1, 0, 2).reshape(-1, nh * vd)
        out = out * sigmoid(self.m.mm(x, get("gate_proj")))
        return self.m.mm(out, get("out_proj"))

    def block(self, l, x, eg, ckv):
        h, m = self.h, self.m
        get = lambda n: self.c.t("decoder.%d" % l, n)
        for s in range(h["num_sites"]):
            if h["sites"][s] == l:
                ek, ev = eg[s]
                alpha = sigmoid((rms_unit(x) * rms_unit(ek)).sum(-1) / np.sqrt(self.m.d))
                x = x + alpha[:, None] * ev
        t = m.attention(zcrms(x, get("norm_in")), zcrms(x, get("norm_in")), get,
                        h["num_heads"], h["num_kv_heads"], True, True, True)
        x = x + sigmoid(get("attn_gate")[0]) * zcrms(t, get("post_attn_norm"))
        t = self.cross(l, zcrms(x, get("cross.cross_norm")), ckv[l])
        x = x + sigmoid(get("cross.cross_gate")[0]) * zcrms(t, get("cross.post_cross_norm"))
        x = x + m.hada_mlp(zcrms(x, get("pre_hada_norm")), get, "mlp.")
        return x

    def logits(self, tokens, ckv):
        """Logits for every position of `tokens` [T, vocab]."""
        emb = self.c.t("decoder", "embedding")
        x = emb[list(tokens)] * np.sqrt(np.float32(self.m.d))
        eg = self.engram_kv(tokens)
        x = self.m.mhc_stack(x, "decoder", self.h["num_layers"],
                             lambda i, u: self.block(i, u, eg, ckv))
        x = zcrms(x, self.c.t("decoder", "final_norm"))
        return x @ emb.T

    def greedy(self, enc, language="en", max_tokens=None):
        """Greedy decoding with the language forced. Returns (text, ids)."""
        ckv = self.cross_kv(enc)
        lang = 8192 + LANGUAGES.index(language)
        toks = [BOS, lang]
        limit = max_tokens or self.h["max_seq_len"]
        while len(toks) < limit:
            nxt = int(np.argmax(self.logits(toks, ckv)[-1]))
            if nxt == EOS:
                break
            toks.append(nxt)
        ids = toks[2:]
        return self.tok.decode(ids), ids


DEFAULTS = dict(
    # front end - PROBED against the engine with the encoder made identity
    gain="peak",                     # clip scaled to peak 1 before the STFT
    center=True, center_pad="nfft", frame_len="nfft", pad_mode="constant",
    periodic=True, power=2.0,
    log="ln_add", log_floor=6e-7,    # ln(mel + 6e-7)
    norm="channel", norm_eps=1e-3, ddof=1,
    # stem - PROBED
    stem_kernel_tf="tf", stem_act="silu", stem_flatten="cf", stem_scale=False,
    pe=True,
    # encoder block - PROBED against the full engine (cos 0.9999)
    enc_rope=True, glu="sigmoid", conv_out_norm="mid", macaron_half=True,
    act_q8=False, kv_q8=False, q_q8=False,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("wav")
    ap.add_argument("--language", default="en")
    a = ap.parse_args()
    m = Whistle(a.blob)
    e = m.encode(load_wav(a.wav))
    print("encoder output", e.shape, file=sys.stderr)
    text, ids = Decoder(m).greedy(e, a.language)
    print("tokens", ids, file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
