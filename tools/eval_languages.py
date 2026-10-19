#!/usr/bin/env python3
"""Colour commands in Whistle's seven languages, scored the way the board runs.

  python3 tools/eval_languages.py model/whistle/whistle_d4.cact ./build/wh_run
  python3 tools/eval_languages.py ... --forced      # tell it the language

Clips: model/whistle/eval/multi (de fr es it nl pl: five phrasings x six
colours, macOS voices, clean and mixed with real INMP441 noise) and
model/whistle/eval/colour (English, the same design). Each clip runs through
host/wh_run with language detection (or --forced), the demo's per-language
keyword lists and the speech gate, and is scored on:

  language   detected language == spoken language
  command    the ACT line has the right colour and flash/solid
  bare word  command accuracy on clips that are just the colour word
"""
import argparse
import collections
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

LANGS = ("en", "de", "fr", "es", "it", "nl", "pl")


def run(args, wav, lang):
    cmd = [args.binary, args.blob, wav, "--led-keywords", "--gate", "12",
           "--lang", lang if args.forced else "auto"] + (["--bonus", str(args.bonus)] if args.bonus else []) \
          + args.extra.split()
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    det = act = None
    for line in r.stderr.splitlines():
        if line.startswith("language "):
            det = line.split()[1]
        elif line.startswith("ACT "):
            act = line
    return r.stdout.strip(), det, act or "ACT none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("binary")
    ap.add_argument("--root", default="model/whistle/eval")
    ap.add_argument("--forced", action="store_true", help="give the true language instead of detecting")
    ap.add_argument("--bonus", type=float, default=0.0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--extra", default="", help="more wh_run arguments, e.g. '--retry 0'")
    a = ap.parse_args()

    clips = []
    multi = json.load(open(os.path.join(a.root, "multi", "refs.json")))
    for k, r in multi.items():
        clips.append((os.path.join(a.root, "multi", k + ".wav"), r))
    eng = json.load(open(os.path.join(a.root, "colour", "refs.json")))
    for k, r in eng.items():
        r = dict(r, lang="en", cond="clean" if k.startswith("clean") else "mic")
        clips.append((os.path.join(a.root, "colour", k + ".wav"), r))

    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda c: run(a, c[0], c[1]["lang"]), clips))

    tot, det_ok = collections.Counter(), collections.Counter()
    cmd_ok, bare_tot, bare_ok = collections.Counter(), collections.Counter(), collections.Counter()
    confusion = collections.defaultdict(collections.Counter)
    misses = collections.defaultdict(list)
    for (wav, r), (text, det, act) in zip(clips, res):
        lang, cond = r["lang"], r["cond"]
        want = "ACT led color=%s mode=%s" % (r["colour"], "flash" if r["template"] == 3 else "solid")
        ok = act.startswith(want)
        tot[lang, cond] += 1
        cmd_ok[lang, cond] += ok
        det_ok[lang] += det == lang
        confusion[lang][det] += 1
        if r["template"] == 2:
            bare_tot[lang] += 1
            bare_ok[lang] += ok
        if not ok:
            misses[lang].append((r["text"], det, text, act))

    print("lang  clips  language  command clean  command mic  bare word   (%s)"
          % ("language forced" if a.forced else "language detected"))
    for lang in LANGS:
        n = tot[lang, "clean"] + tot[lang, "mic"]
        if not n:
            continue
        print("%-4s  %5d  %7.1f%%  %12.1f%%  %10.1f%%  %8.1f%%" % (
            lang, n, 100 * det_ok[lang] / n,
            100 * cmd_ok[lang, "clean"] / tot[lang, "clean"],
            100 * cmd_ok[lang, "mic"] / tot[lang, "mic"],
            100 * bare_ok[lang] / max(1, bare_tot[lang])))
    if not a.forced:
        print("\ndetected-language confusion (spoken: detected counts)")
        for lang in LANGS:
            print("  %s: %s" % (lang, dict(confusion[lang].most_common())))
    print("\nexample misses per language:")
    for lang in LANGS:
        for said, det, text, act in misses[lang][:4]:
            print("  %s  %-48s [%s] %-40s %s" % (lang, said[:48], det, text[:40], act[4:]))
    if a.out:
        json.dump([dict(r, wav=w, hyp=t, det=d, act=c) for (w, r), (t, d, c) in zip(clips, res)],
                  open(a.out, "w"), ensure_ascii=False, indent=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
