#!/usr/bin/env python3
"""Accuracy of Whistle at each decoder depth, to choose a flash layout.

  python3 tools/eval_whistle.py model/whistle/whistle.cact \\
      --engine model/needle3-engine/needle      # official CLI (--audio-depth)
  python3 tools/eval_whistle.py model/whistle/whistle.cact --ours ./build/wh_run

Test sets (built by hand, not included): model/whistle/eval/libri
(73 LibriSpeech utterances, refs.json) and model/whistle/eval/colour (the LED
tool's six colours in five phrasings and six voices, clean and mixed with
real INMP441 noise at a typical speaking level).

Scores: WER after lowercasing and stripping punctuation, and for the colour
set the fraction of clips whose transcript names the right colour and no
other - the thing the LED actually depends on.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

COLOURS = ("red", "green", "blue", "yellow", "purple", "white")


def norm(s):
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


def transcribe(args, wav, depth):
    env = dict(os.environ, NEEDLE_TELEMETRY="0")
    if args.ours:
        cmd = [args.ours, args.blob, wav] + (["--depth", str(depth)] if depth else []) + args.extra.split()
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        act = [l for l in r.stderr.splitlines() if l.startswith("ACT ")]
        return r.stdout.strip() + ("\t" + act[-1] if act else "")
    cmd = [args.engine, "--model", args.blob, "--audio", wav, "--audio-language", "en",
           "--audio-depth", str(depth), "--threads", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    for line in reversed(out.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line).get("text", "")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--engine", help="official needle CLI")
    g.add_argument("--ours", help="host/wh_run")
    ap.add_argument("--depths", default="2,3,4,5,6,7,8")
    ap.add_argument("--root", default="model/whistle/eval")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", help="write every transcript to this JSON")
    ap.add_argument("--extra", default="", help="more wh_run arguments, e.g. '--gate 12 --keywords red,blue'")
    a = ap.parse_args()

    sets = {}
    for name in ("libri", "colour"):
        refs = json.load(open(os.path.join(a.root, name, "refs.json")))
        sets[name] = {k: (os.path.join(a.root, name, k + ".wav"),
                          v if isinstance(v, dict) else {"text": v}) for k, v in refs.items()}
    depths = [int(d) for d in a.depths.split(",")]   # 0 = the blob's own depth
    jobs = [(s, k, d) for s in sets for k in sets[s] for d in depths]
    with ThreadPoolExecutor(a.jobs) as ex:
        hyps = list(ex.map(lambda j: transcribe(a, sets[j[0]][j[1]][0], j[2]), jobs))
    res = {(s, k, d): h for (s, k, d), h in zip(jobs, hyps)}

    print("depth   libri WER   colour clean: acc  WER   colour mic noise: acc  WER"
          + ("   command (colour+mode) clean / mic" if a.ours else ""))
    table = {}
    for d in depths:
        row = {}
        e = n = 0
        for k, (_, r) in sets["libri"].items():
            ref = norm(r["text"])
            e += edits(ref, norm(res[("libri", k, d)].split("\t")[0]))
            n += len(ref)
        row["libri_wer"] = e / n
        for cond in ("clean", "mic"):
            ok = e = n = 0
            items = [(k, r) for k, (_, r) in sets["colour"].items() if k.startswith(cond)]
            cmd_ok = 0
            for k, r in items:
                text, _, act = res[("colour", k, d)].partition("\t")
                hyp = norm(text)
                want = "ACT led color=%s mode=%s" % (r["colour"], "flash" if r["template"] == 3 else "solid")
                cmd_ok += act.startswith(want)
                said = {c for c in COLOURS if c in hyp}
                ok += said == {r["colour"]}
                ref = norm(r["text"])
                e += edits(ref, hyp)
                n += len(ref)
            row[cond + "_acc"], row[cond + "_wer"] = ok / len(items), e / n
            row[cond + "_cmd"] = cmd_ok / len(items)
        table[d] = row
        print("%5d   %8.1f%%   %15.1f%% %5.1f%%   %19.1f%% %5.1f%%" % (
            d, 100 * row["libri_wer"], 100 * row["clean_acc"], 100 * row["clean_wer"],
            100 * row["mic_acc"], 100 * row["mic_wer"])
            + ("   %21.1f%% / %.1f%%" % (100 * row["clean_cmd"], 100 * row["mic_cmd"]) if a.ours else ""))

    # Which colour clips fail, at the shallowest and the full depth.
    for d in (depths[0], depths[-1]):
        bad = [(k, sets["colour"][k][1], res[("colour", k, d)]) for k in sets["colour"]
               if {c for c in COLOURS if c in norm(res[("colour", k, d)].split("\t")[0])}
               != {sets["colour"][k][1]["colour"]}]
        print("\ndepth %d colour misses (%d):" % (d, len(bad)))
        for k, r, h in bad[:25]:
            print("  %-10s %-22s %-28r -> %r" % (k, r["voice"][:22], r["text"], h))
    if a.out:
        json.dump({"%s/%s/%d" % j: h for j, h in res.items()}, open(a.out, "w"), indent=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
