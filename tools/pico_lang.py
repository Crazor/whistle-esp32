#!/usr/bin/env python3
"""Fetch SVOX Pico lingware (Apache 2.0) for the text to speech.

    tools/pico_lang.py                      # en de fr es it -> model/pico/
    tools/pico_lang.py en de --out DIR      # just these

Each language is a text analysis file (*_ta.bin) and a signal generation file
(*_sg.bin), 0.9-1.5 MB together. They come from AOSP's external/svox at a
pinned commit and are checked against the hashes below, so a changed upstream
cannot slip in. Files already present and correct are skipped.
"""
import argparse
import base64
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request

COMMIT = "c5a33e4fdc55b63d4aa33aecefcc91b63da921e7"
URL = ("https://android.googlesource.com/platform/external/svox/+/"
       + COMMIT + "/pico/lang/{}?format=TEXT")   # gitiles serves base64

FILES = {
    "en": {"en-US_ta.bin": "3a2a4d23de2a44dbbff50f69dcb9486be53b6d0a871d77cdc35f45c3e25e82aa",
           "en-US_lh0_sg.bin": "d6946526696d9eb085f7b23b1fbdf7a9986caca658b645e9eb73c6514fd4789d"},
    "en-GB": {"en-GB_ta.bin": "57998f6423236dea25cd572a73b13f371e64155840e31b571eba7fca0b521b7e",
              "en-GB_kh0_sg.bin": "8f9f5d13a920a9f3371517178d42ce89b835f1e587760c2db280beb593c388fa"},
    "de": {"de-DE_ta.bin": "df292826754713d5efb6c95d60582c17753e95856b4c735c3150db722b2c218a",
           "de-DE_gl0_sg.bin": "39147b816d08bbbe64b23797804cf691a761ab73a7689ca9e2b6f87ace722654"},
    "fr": {"fr-FR_ta.bin": "d9da5eb3cd7aac7071ab8ff7fee8d716f58572ee5393c1f0a8db946d6bf32f97",
           "fr-FR_nk0_sg.bin": "85e2d63cf9f5522ea2a483d25ddd1cb0c6bfeecc39c92884aecfe7d3f179837a"},
    "es": {"es-ES_ta.bin": "b18c60baf249466bf8785d333d8cb4cf1f6b62e5545986f621c66568f1851961",
           "es-ES_zl0_sg.bin": "82e724d3c788767ec95f0d22b0ab738e677ad397f26ec0bbbf2c07aafdf29000"},
    "it": {"it-IT_ta.bin": "9b243c3470660293691d0d4e000901d7c5d0fa71976b33986e92e93a860ade33",
           "it-IT_cm0_sg.bin": "c48be4dc1f5a3c77a3dc4c8a457949b69bc433d674fac4fac91d49c01770e7e4"},
}
DEFAULT = ["en", "de", "fr", "es", "it"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch(name):
    """gitiles answers the odd request with a 503; try again before giving up."""
    for attempt in range(4):
        try:
            with urllib.request.urlopen(URL.format(name), timeout=60) as r:
                return base64.b64decode(r.read())
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == 3:
                raise
        time.sleep(2 << attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("langs", nargs="*", default=DEFAULT, help=" ".join(FILES))
    ap.add_argument("--out", default="model/pico")
    args = ap.parse_args()

    bad = [l for l in args.langs if l not in FILES]
    if bad:
        sys.exit("no Pico voice for %s (have %s)" % (", ".join(bad), " ".join(FILES)))
    os.makedirs(args.out, exist_ok=True)
    for lang in args.langs:
        for name, want in FILES[lang].items():
            path = os.path.join(args.out, name)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    if sha256(f.read()) == want:
                        continue
            data = fetch(name)
            if sha256(data) != want:
                sys.exit("%s: hash mismatch, not written" % name)
            with open(path, "wb") as f:
                f.write(data)
            print("%s  %7.1f KB" % (path, len(data) / 1024))


if __name__ == "__main__":
    main()
