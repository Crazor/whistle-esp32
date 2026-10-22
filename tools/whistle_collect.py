#!/usr/bin/env python3
"""Record phrases through esp32/whistle_stt for tools/whistle_bench.py.

    python tools/whistle_collect.py PORT SESSION_DIR [--reset] [--all]

Walks through SESSION_DIR/phrases.txt (one prompt per line; `#` starts a
comment, and a `# lang: de` line names the session's language). For each
prompt: hold BOOT, say it, release. The board transcribes it and sends back
the exact samples it transcribed (`dump on`), which are saved as NNN.wav
(16 kHz mono s16le) with a line in SESSION_DIR/index.jsonl: the file, the
prompt as reference, the board's transcript, the noise floor, the depth and
language at capture, and the time.

Type a key and Enter:
    (Enter)   keep the take and go on        r   discard the take, say it again
    s         skip this prompt               f   record a free phrase, then
    q         quit                               type what you said
Pressing BOOT again before Enter replaces the take. Any other line (such as
`set depth=4` or `set lang=de`) goes to the board.

Prompts already in index.jsonl are skipped, so a session can be continued;
--all asks for every prompt again. Clips are recordings of your voice: the
clips/ directory is in .gitignore.
"""
import argparse, datetime, json, os, re, select, sys, wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from whistle_serial import Board, BoardError, Capture, SAMPLE_RATE  # noqa: E402

HELP = ('  hold BOOT and speak | Enter keep  r redo  s skip  f free phrase  '
        'q quit  set ... to the board')


def read_phrases(path):
    phrases, lang = [], None
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if line.startswith('#'):
                m = re.fullmatch(r'#\s*lang:\s*(\w+)\s*', line)
                if m:
                    lang = m.group(1)
                continue
            if line:
                phrases.append(line)
    return phrases, lang


def read_index(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


class Session:
    def __init__(self, d, lang):
        self.dir, self.lang = d, lang
        self.index = os.path.join(d, 'index.jsonl')
        nums = [int(m.group(1)) for f in os.listdir(d)
                if (m := re.fullmatch(r'(\d{3,})\.wav', f))]
        self.next = max(nums, default=0) + 1

    def save(self, take, ref):
        name = f'{self.next:03d}.wav'
        self.next += 1
        with wave.open(os.path.join(self.dir, name), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(take['pcm'])
        res = take['res']
        entry = {
            'file': name,
            'ref': ref,
            'hyp': res.get('text', ''),
            'floor_dbfs': take['floor_dbfs'],
            'samples': len(take['pcm']) // 2,
            'depth': int(res['depth']) if 'depth' in res else None,
            'lang': res.get('lang'),
            'lang_detected': res.get('lang_detected'),
            'session_lang': self.lang,
            'time': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        }
        with open(self.index, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f'  saved {name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('port')
    ap.add_argument('session_dir')
    ap.add_argument('--reset', action='store_true', help='restart the firmware first')
    ap.add_argument('--all', action='store_true', help='ask for prompts already recorded too')
    a = ap.parse_args()

    phrases, lang = read_phrases(os.path.join(a.session_dir, 'phrases.txt'))
    sess = Session(a.session_dir, lang)
    if not a.all:
        done = {e['ref'] for e in read_index(sess.index)}
        left = [p for p in phrases if p not in done]
        if len(left) < len(phrases):
            print(f'{len(phrases) - len(left)} of {len(phrases)} prompts already recorded; '
                  'skipping them (--all to record them again)')
        phrases = left

    def show(line):
        if not line.startswith('PCM'):
            print('  | ' + line, flush=True)

    board = Board(a.port, reset=a.reset, echo=show)
    try:
        board.dump(True)
        cfg = board.cfg()
    except BoardError as e:
        sys.exit(f'{e}. Is esp32/whistle_stt running on {a.port}? (--reset restarts it)')
    print(f'board: depth={cfg.get("depth")} lang={cfg.get("lang")} '
          f'model_layers={cfg.get("model_layers")}')
    print(HELP)

    cap = Capture()
    i, take, free = 0, None, False

    def prompt():
        if free:
            print('\n>> free phrase: hold BOOT and say anything', flush=True)
        elif i < len(phrases):
            print(f'\n>> [{i + 1}/{len(phrases)}] {phrases[i]}', flush=True)
        else:
            print('\nAll prompts done. f for a free phrase, q to quit.', flush=True)

    prompt()
    try:
        while True:
            line = board.get(0.05)
            while line is not None:
                show(line)
                got = cap.feed(line)
                if got and got.get('too_short'):
                    print('  too short: hold BOOT while you speak')
                elif got:
                    res = got['res']
                    hyp = res.get('text', '')
                    print(f'  heard: {hyp!r}' if hyp else '  heard nothing')
                    if free:
                        ref = input('  what did you say? (empty: discard) ').strip()
                        if ref:
                            sess.save(got, ref)
                        free = False
                        prompt()
                    elif i < len(phrases):
                        take = got
                        print('  Enter keeps it, r says it again')
                    else:
                        print('  not saved: f first for a free phrase')
                line = board.get(0)

            if not select.select([sys.stdin], [], [], 0)[0]:
                continue
            cmd = sys.stdin.readline()
            if not cmd:
                break                               # end of input
            cmd = cmd.strip()
            if cmd == 'q':
                if take:
                    sess.save(take, phrases[i])
                break
            if take is not None:
                if cmd == '':
                    sess.save(take, phrases[i])
                    take, i = None, i + 1
                    prompt()
                elif cmd == 'r':
                    take = None
                    print('  discarded')
                    prompt()
                else:
                    print('  Enter keeps the take, r discards it')
            elif cmd == 's' and i < len(phrases):
                i += 1
                prompt()
            elif cmd == 'f':
                free = True
                prompt()
            elif cmd == '':
                print(HELP)
                prompt()
            elif cmd in ('r', 's'):
                print('  nothing to ' + ('redo' if cmd == 'r' else 'skip'))
            else:
                board.send(cmd)
    except KeyboardInterrupt:
        print('\nstopped' + ('; the last take was not saved' if take else ''))
    finally:
        try:
            board.send('dump off')
        finally:
            board.close()


if __name__ == '__main__':
    main()
