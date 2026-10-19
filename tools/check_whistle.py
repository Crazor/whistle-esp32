#!/usr/bin/env python3
"""Validate the Whistle C engine against the NumPy reference, and both against
the official engine (pip install cactus-needle).

  python3 tools/check_whistle.py model/whistle/whistle.cact ./build/wh_run \\
      model/whistle/wav/*.wav

Three layers, as check_engine.py / ref_forward.py do for Needle 2:

  1. C vs NumPy, encoder output             tight: same maths, float32
  2. C vs NumPy, logits over a token prefix tight: the C decoder is
     incremental (KV cache, Q/K/V tap and engram histories) while the
     reference recomputes the whole sequence, so agreement validates both
  3. NumPy encoder vs needle.Whistle().embed and C transcripts vs
     needle.transcribe(language="en")

Tolerances for (3) are looser and per frame. The official engine's numerics
are not published, and near-silent frames are ill-conditioned: the last
encoder layer amplifies tiny differences there (the engine is as sensitive to
its own rounding), so a pause can differ by 20%+ while every speech frame
agrees to ~1%. The check therefore scores frames that carry signal, and
transcripts, which are what the decoder sees. It also reports the engine's
own 5-beam vs our greedy decoding when transcripts differ.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

from whistle_ref import BOS, Decoder, Whistle, load_wav

TIGHT_ENC = 2e-4      # C vs NumPy encoder, relative
TIGHT_LOGITS = 2e-3   # C vs NumPy logits, max abs over the vocabulary
ENGINE_FRAME = 0.05   # NumPy vs official, per speech frame (relative)


def run_c(binary, blob, wav, tokens=None):
    with tempfile.TemporaryDirectory() as d:
        enc_p, lg_p = os.path.join(d, "enc.f32"), os.path.join(d, "lg.f32")
        # --float: this check is about the maths, so it compares the float
        # matmul path; the int8 path is scored on transcripts (eval_whistle).
        cmd = [binary, blob, wav, "--float", "--enc-out", enc_p]
        if tokens:
            cmd += ["--logits-out", lg_p, "--tokens", ",".join(map(str, tokens))]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        enc = np.fromfile(enc_p, np.float32).reshape(-1, 512)
        lg = np.fromfile(lg_p, np.float32).reshape(len(tokens), -1) if tokens else None
        return out.stdout.strip(), enc, lg, out.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("binary")
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--no-engine", action="store_true",
                    help="skip the comparisons against the official package")
    a = ap.parse_args()

    m = Whistle(a.blob)
    dec = Decoder(m)
    W = None
    if not a.no_engine:
        os.environ["NEEDLE_TELEMETRY"] = "0"   # the package reports usage by default
        import needle
        W = needle.Whistle(weights=a.blob)

    fails = 0
    for wav in a.wavs:
        name = os.path.basename(wav)
        x = load_wav(wav)
        ref_enc = m.encode(x)
        text_ref, ids_ref = dec.greedy(ref_enc, "en")
        prefix = [BOS, 8192] + ids_ref[:6]
        text_c, enc_c, lg_c, err = run_c(a.binary, a.blob, wav, prefix)

        enc_rel = float(np.linalg.norm(enc_c - ref_enc) / np.linalg.norm(ref_enc))
        lg_ref = dec.logits(prefix, dec.cross_kv(ref_enc))
        lg_diff = float(np.abs(lg_c - lg_ref).max())
        same_argmax = bool((lg_c.argmax(1) == lg_ref.argmax(1)).all())
        ok1 = enc_rel < TIGHT_ENC
        ok2 = lg_diff < TIGHT_LOGITS and same_argmax
        print("%s  (%.2f s, %d frames)" % (name, len(x) / 16000, ref_enc.shape[0]))
        print("  C vs NumPy  encoder rel %.2e %s | logits max|d| %.2e over %d positions, argmax %s %s"
              % (enc_rel, "ok" if ok1 else "FAIL", lg_diff, len(prefix),
                 "same" if same_argmax else "DIFFERS", "ok" if ok2 else "FAIL"))
        print("  transcript C     : %s" % text_c)
        if text_c != text_ref:
            print("  transcript NumPy : %s   (C and NumPy disagree: FAIL)" % text_ref)
        fails += (not ok1) + (not ok2) + (text_c != text_ref)

        if W is not None:
            eng = np.array(W.embed(x.tolist())).reshape(-1, 512)
            pf = np.linalg.norm(ref_enc - eng, axis=1) / np.linalg.norm(eng, axis=1)
            # Frames with signal: 80 ms windows within 20 dB of the clip's
            # loudest window. Quieter windows are pauses: the INMP441's noise
            # floor sits only ~27 dB below speech, TTS has digital silence,
            # and either way those frames are the ill-conditioned ones.
            seg = np.array([np.sqrt(np.mean(x[i * 1280:(i + 1) * 1280] ** 2))
                            for i in range(len(pf))])
            speech = seg > 0.1 * seg.max()
            worst = float(pf[speech].max()) if speech.any() else 0.0
            ok3 = worst < ENGINE_FRAME
            t_eng = W.transcribe(x.tolist(), language="en")["text"]
            print("  NumPy vs engine encoder: speech frames %d/%d, median %.3f, worst %.3f %s; "
                  "pause frames worst %.3f (not scored)"
                  % (speech.sum(), len(pf), float(np.median(pf[speech])) if speech.any() else 0,
                     worst, "ok" if ok3 else "FAIL",
                     float(pf[~speech].max()) if (~speech).any() else 0.0))
            same = t_eng == text_c
            print("  transcript engine: %s  [%s]" % (
                t_eng, "match" if same else
                "differs (engine runs 5-beam search, this is greedy)" if t_eng.lower() == text_c.lower()
                else "DIFFERS"))
            fails += (not ok3)
        print()

    print("problems: %d" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
