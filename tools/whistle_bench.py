#!/usr/bin/env python3
"""Replay recorded sessions on esp32/whistle_stt at several decoder depths and
language settings, and score them.

    python tools/whistle_bench.py PORT SESSION_DIR... [--depth 3,4,8]
        [--lang de,auto] [--out results.csv] [--reset]
    python tools/whistle_bench.py --summary results.csv

Sessions come from tools/whistle_collect.py: SESSION_DIR/index.jsonl and its
WAVs. Each clip is uploaded once (`clip`), then transcribed (`run`) under
every setting, so every result is measured on the board: one CSV row per run,
written as it comes in. --lang takes settings separated by spaces or `;` when
one of them is a list (`--lang "de;de,en;auto"`).

The summary per setting gives the word and character error rates over all
clips (both texts lowercased, punctuation removed; CER catches German
compounds split in two, which WER counts as two wrong words), the median and
interquartile range of the total time, encoding time per second of speech,
decoding time per token, and how often the detected language is the
session's (`# lang:` in phrases.txt, else --session-lang).
"""
import argparse, csv, os, re, statistics, sys, time, wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from whistle_serial import (Board, BoardError, SAMPLE_RATE, edits,  # noqa: E402
                            normalise)
from whistle_collect import read_index  # noqa: E402

SECS_PER_RUN = 4.0        # board time per run, plus the upload once per clip

FIELDS = ['session', 'file', 'depth', 'lang', 'ref', 'hyp', 'session_lang',
          'lang_detected', 'audio_ms', 'speech_ms', 'encode_ms', 'decode_ms',
          'total_ms', 'tokens', 'word_errors', 'ref_words', 'char_errors', 'ref_chars']


def load_clips(dirs, default_lang):
    clips = []
    for d in dirs:
        idx = os.path.join(d, 'index.jsonl')
        if not os.path.exists(idx):
            sys.exit(f'{idx} not found: record the session with tools/whistle_collect.py')
        for e in read_index(idx):
            with wave.open(os.path.join(d, e['file']), 'rb') as w:
                if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SAMPLE_RATE):
                    sys.exit(f'{d}/{e["file"]}: not 16 kHz mono s16le')
                pcm = w.readframes(w.getnframes())
            clips.append({'session': os.path.basename(os.path.normpath(d)),
                          'file': e['file'], 'ref': e['ref'], 'pcm': pcm,
                          'floor_dbfs': e.get('floor_dbfs'),
                          'session_lang': e.get('session_lang') or default_lang})
    return clips


def score(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    return {'word_errors': edits(r.split(), h.split()), 'ref_words': len(r.split()),
            'char_errors': edits(r, h), 'ref_chars': len(r)}


def spread(xs):
    """'median (p25-p75)' of a list of numbers."""
    if not xs:
        return '-'
    xs = sorted(xs)
    if len(xs) < 2:
        return f'{xs[0]:.0f}'
    q = statistics.quantiles(xs, n=4, method='inclusive')
    return f'{statistics.median(xs):.0f} ({q[0]:.0f}-{q[2]:.0f})'


def summarise(rows):
    groups = {}
    for r in rows:
        groups.setdefault((int(r['depth']), r['lang']), []).append(r)
    head = ['depth', 'lang', 'clips', 'WER', 'CER', 'total ms', 'enc ms/s speech',
            'dec ms/token', 'lang ok']
    table = [head]
    for (depth, lang), rs in sorted(groups.items()):
        num = lambda k: [float(r[k]) for r in rs]  # noqa: E731
        we, rw = sum(num('word_errors')), sum(num('ref_words'))
        ce, rc = sum(num('char_errors')), sum(num('ref_chars'))
        enc = [float(r['encode_ms']) / (float(r['speech_ms']) / 1000)
               for r in rs if float(r['speech_ms']) > 0]
        dec = [float(r['decode_ms']) / float(r['tokens']) for r in rs if float(r['tokens']) > 0]
        ok = sum(r['lang_detected'] == r['session_lang'] for r in rs)
        table.append([str(depth), lang, str(len(rs)),
                      f'{100 * we / rw:.1f}%' if rw else '-',
                      f'{100 * ce / rc:.1f}%' if rc else '-',
                      spread(num('total_ms')), spread(enc), spread(dec),
                      f'{100 * ok / len(rs):.0f}%'])
    w = [max(len(row[i]) for row in table) for i in range(len(head))]
    lines = ['  '.join(c.ljust(w[i]) for i, c in enumerate(row)).rstrip() for row in table]
    lines.insert(1, '  '.join('-' * x for x in w))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('port', nargs='?')
    ap.add_argument('sessions', nargs='*')
    ap.add_argument('--depth', default='3,4,8')
    ap.add_argument('--lang', default='de,auto',
                    help='settings, comma-separated; use ; or spaces when a setting is a list')
    ap.add_argument('--session-lang', default='de',
                    help='language of sessions whose phrases.txt names none')
    ap.add_argument('--out', default='results.csv')
    ap.add_argument('--reset', action='store_true', help='restart the firmware first')
    ap.add_argument('--summary', metavar='CSV', help='only summarise an earlier run')
    a = ap.parse_args()

    if a.summary:
        with open(a.summary, newline='', encoding='utf-8') as f:
            print(summarise(list(csv.DictReader(f))))
        return
    if not a.port or not a.sessions:
        ap.error('PORT and at least one SESSION_DIR are needed')

    depths = [int(d) for d in a.depth.split(',')]
    sep = ';' if ';' in a.lang else (' ' if ' ' in a.lang.strip() else ',')
    langs = [l.strip() for l in a.lang.split(sep) if l.strip()]
    settings = [(d, l) for d in depths for l in langs]
    clips = load_clips(a.sessions, a.session_lang)
    if not clips:
        sys.exit('no clips in ' + ' '.join(a.sessions))

    board = Board(a.port, reset=a.reset)
    try:
        cfg = board.cfg()
    except BoardError as e:
        sys.exit(f'{e}. Is esp32/whistle_stt running on {a.port}? (--reset restarts it)')
    layers = int(cfg['model_layers'])
    if max(depths) > layers or min(depths) < 2:
        sys.exit(f'the flashed model has {layers} decoder layers, so depth must be 2..{layers} '
                 f'(asked for {a.depth}). Depths above 4 need the full whistle.cact '
                 '(see the README).')
    for l in langs:
        try:
            board.cfg(lang=l)
        except BoardError as e:
            sys.exit(f'lang={l}: {e}')

    runs = len(clips) * len(settings)
    secs = runs * SECS_PER_RUN
    print(f'{len(clips)} clips x {len(settings)} settings = {runs} runs, '
          f'about {secs / 60:.0f} min', flush=True)

    rows, t0 = [], time.time()
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        out = csv.DictWriter(f, FIELDS)
        out.writeheader()
        for ci, c in enumerate(clips):
            samples = memoryview(c['pcm']).cast('h')      # WAV: little-endian
            board.upload(samples, c['floor_dbfs'])
            for depth, lang in settings:
                board.cfg(depth=depth, lang=lang)
                res = board.run()
                row = {'session': c['session'], 'file': c['file'], 'depth': depth,
                       'lang': lang, 'ref': c['ref'], 'hyp': res.get('text', ''),
                       'session_lang': c['session_lang'],
                       'lang_detected': res.get('lang_detected', '')}
                for k in ('audio_ms', 'speech_ms', 'encode_ms', 'decode_ms', 'total_ms',
                          'tokens'):
                    row[k] = res.get(k, '')
                row.update(score(c['ref'], row['hyp']))
                out.writerow(row)
                f.flush()
                rows.append(row)
                print(f'[{len(rows)}/{runs}] {c["session"]}/{c["file"]} depth={depth} '
                      f'lang={lang} {row["total_ms"]} ms  {row["hyp"]!r}', flush=True)
            left = (time.time() - t0) / (ci + 1) * (len(clips) - ci - 1)
            if ci + 1 < len(clips):
                print(f'  about {left / 60:.1f} min left', flush=True)
    board.close()
    print(f'\n{a.out}: {len(rows)} runs in {(time.time() - t0) / 60:.1f} min\n')
    print(summarise(rows))


if __name__ == '__main__':
    main()
