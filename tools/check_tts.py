#!/usr/bin/env python3
"""Round trip: Pico says each demo command, Whistle transcribes it back.

  python3 tools/check_tts.py model/whistle/whistle_d4.cact
      [--tts ./build/tts_run --asr ./build/wh_run --lang-dir model/pico]

A listening test that needs no ears: if Whistle hears the colour Pico meant,
in the language Pico spoke, the voice is intelligible. Prints each transcript,
the detected language and the LED action, and per language the word error
rate, how many commands came back with the right colour (the exit status) and
how often Whistle detected the language Pico spoke.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

# The README's example commands, two per language, with the colour each names.
PHRASES = {
    "en": [("Flash the light blue.", "blue"), ("Turn the light green for five seconds.", "green")],
    "de": [("Lass das Licht rot blinken.", "red"), ("Mach das Licht gelb.", "yellow")],
    "fr": [("Fais clignoter la lumière en violet.", "purple"), ("Mets la lumière en vert.", "green")],
    "es": [("Pon la luz amarilla.", "yellow"), ("Haz parpadear la luz roja.", "red")],
    "it": [("Fai lampeggiare la luce verde.", "green"), ("Metti la luce blu.", "blue")],
}


def norm(s):
    return re.sub(r"[^\w' ]+", " ", s.lower()).split()


def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("blob", help="Whistle .cact")
    ap.add_argument("--tts", default="./build/tts_run")
    ap.add_argument("--asr", default="./build/wh_run")
    ap.add_argument("--lang-dir", default="model/pico")
    args = ap.parse_args()

    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for lang, phrases in PHRASES.items():
            words = errs = right = same = 0
            for i, (text, colour) in enumerate(phrases):
                wav = os.path.join(tmp, "%s%d.wav" % (lang, i))
                r = subprocess.run([args.tts, args.lang_dir, lang, text, wav], capture_output=True, text=True)
                if r.returncode:
                    sys.exit("tts_run: " + r.stderr.strip())
                r = subprocess.run([args.asr, args.blob, wav, "--lang", "auto", "--led-keywords"],
                                   capture_output=True, text=True, errors="replace")
                heard = r.stdout.strip()
                act = next((l for l in r.stderr.splitlines() if l.startswith("ACT ")), "ACT none")
                det = next((l.split()[1] for l in r.stderr.splitlines() if l.startswith("language ")), "?")
                ok = ("color=%s " % colour) in act
                right += ok
                same += det == lang
                words += len(norm(text))
                errs += edits(norm(text), norm(heard))
                print("%s %s  %-44s -> %-44s [%s] %s" % (lang, "ok  " if ok else "MISS", text, heard, det, act))
            failed += len(phrases) - right
            print("%s  WER %.0f%%  right colour %d/%d  language detected %d/%d\n"
                  % (lang, 100.0 * errs / words, right, len(phrases), same, len(phrases)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
