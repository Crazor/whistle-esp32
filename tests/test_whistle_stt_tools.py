"""whistle_collect.py / whistle_bench.py against a simulated whistle_stt
console; no board or pyserial required."""
import base64, csv, io, json, os, sys, tempfile, threading, unittest, wave
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
import whistle_bench as bench  # noqa: E402
import whistle_serial as ws  # noqa: E402


class FakeBoard:
    """The serial side of esp32/whistle_stt's console, as far as the tools use
    it. `run` "transcribes" a clip as the text registered for its length."""

    def __init__(self, layers=8, texts=None):
        self.layers, self.texts = layers, texts or {}
        self.depth, self.lang, self.dump = layers, 'auto', False
        self.out, self.cv = bytearray(), threading.Condition()
        self.inbuf, self.clip, self.want, self.uploading = b'', None, 0, False
        self.runs = []

    # pyserial's side
    @property
    def in_waiting(self):
        return len(self.out)

    def read(self, n):
        with self.cv:
            self.cv.wait_for(lambda: self.out, timeout=0.05)
            b, self.out[:] = bytes(self.out[:n]), self.out[n:]
            return b

    def write(self, b):
        self.inbuf += b
        while b'\n' in self.inbuf:
            line, self.inbuf = self.inbuf.split(b'\n', 1)
            self.command(line.decode())

    def close(self):
        pass

    # the firmware's side
    def say(self, line):
        with self.cv:
            self.out += (line + '\r\n').encode()
            self.cv.notify_all()

    def cfg(self):
        self.say(f'CFG depth={self.depth} layers=0,7 lang={self.lang} '
                 f'model_layers={self.layers} dump={"on" if self.dump else "off"}')

    def res(self, n, text):
        esc = text.replace('\\', '\\\\').replace('"', '\\"')
        self.say(f'RES depth={self.depth} lang={self.lang} lang_detected=de '
                 f'audio_ms={n // 16} speech_ms={n // 16} encode_ms=900 decode_ms=1200 '
                 f'total_ms=2100 tokens=10 text="{esc}"')

    def command(self, line):
        if self.uploading:
            if line == 'clip end':
                self.uploading = False
                ok = len(self.clip) == 2 * self.want
                self.say(f'OK clip n={self.want}' if ok else 'ERR clip short')
            else:
                self.clip += base64.b64decode(line, validate=True)
                self.say('OK')
            return
        words = line.split()
        if words[0] == 'set':
            for kv in words[1:]:
                k, v = kv.split('=')
                if k == 'depth':
                    if not 2 <= int(v) <= self.layers:
                        return self.say(f'ERR set depth must be 2..{self.layers}')
                    self.depth = int(v)
                else:
                    self.lang = v
            self.cfg()
        elif words[0] == 'dump':
            self.dump = words[1] == 'on'
            self.cfg()
        elif words[0] == 'clip':
            kv = dict(w.split('=') for w in words[1:])
            self.want, self.clip, self.uploading = int(kv['n']), b'', True
            self.floor = kv['floor_dbfs']
            self.say('OK')
        elif words[0] == 'run':
            n = len(self.clip) // 2
            self.runs.append((self.depth, self.lang, self.floor, self.clip))
            self.say('EVT transcribing')
            self.say('TXT ' + self.texts.get(n, ''))
            self.res(n, self.texts.get(n, ''))

    def press(self, pcm, text, floor='-84.1234'):
        """A live clip with `dump on`."""
        self.say('EVT recording')
        self.say('EVT released held_ms=2000')
        self.res(len(pcm) // 2, text)
        for i in range(0, len(pcm), 96):
            self.say('PCM ' + base64.b64encode(pcm[i:i + 96]).decode())
        self.say(f'PCM end rate=16000 ch=1 n={len(pcm) // 2} floor_dbfs={floor}')


def tone(n, seed=1):
    return b''.join(((i * seed * 37) % 2000 - 1000).to_bytes(2, 'little', signed=True)
                    for i in range(n))


class ProtocolTests(unittest.TestCase):
    def test_parse_kv_unescapes_text(self):
        kv = ws.parse_kv(r'RES depth=4 lang=de,en text="er sagte \"hallo\" \\ ja"')
        self.assertEqual(kv['lang'], 'de,en')
        self.assertEqual(kv['text'], 'er sagte "hallo" \\ ja')

    def test_upload_and_run(self):
        fb = FakeBoard(texts={1000: 'Mach das Licht an.'})
        b = ws.Board('x', ser=fb)
        try:
            pcm = tone(1000)
            b.upload(memoryview(pcm).cast('h'), -83.5)
            res = b.run()
        finally:
            b.close()
        self.assertEqual(fb.clip, pcm)
        self.assertEqual(fb.floor, '-83.5000')
        self.assertEqual(res['text'], 'Mach das Licht an.')

    def test_error_reply_raises(self):
        b = ws.Board('x', ser=FakeBoard(layers=4))
        try:
            with self.assertRaisesRegex(ws.BoardError, 'depth must be 2..4'):
                b.cfg(depth=8)
        finally:
            b.close()

    def test_capture(self):
        fb, cap, got = FakeBoard(), ws.Capture(), []
        pcm = tone(777)
        fb.press(pcm, 'Hallo Welt.')
        for line in fb.out.decode().split('\r\n'):
            if line and (r := cap.feed(line)):
                got.append(r)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]['pcm'], pcm)
        self.assertEqual(got[0]['floor_dbfs'], -84.1234)
        self.assertEqual(got[0]['res']['text'], 'Hallo Welt.')
        self.assertEqual(cap.feed('EVT too_short'), {'too_short': True})


class ScoreTests(unittest.TestCase):
    def test_normalise(self):
        self.assertEqual(ws.normalise('Schalte das  Licht ein!'), 'schalte das licht ein')
        self.assertEqual(ws.normalise('Grüß Gott, Straße.'), 'grüß gott straße')

    def test_compound_split_is_harsh_on_wer_mild_on_cer(self):
        s = bench.score('Das Brautkleid.', 'das Braut Kleid')
        self.assertEqual((s['word_errors'], s['ref_words']), (2, 2))
        self.assertEqual((s['char_errors'], s['ref_chars']), (1, 14))

    def test_summary(self):
        rows = [dict(depth='4', lang='de', session_lang='de', lang_detected='de',
                     word_errors=1, ref_words=4, char_errors=2, ref_chars=20,
                     encode_ms=900, speech_ms=1000, decode_ms=1300, tokens=10,
                     total_ms=t) for t in (2000, 2200, 2400)]
        text = bench.summarise(rows)
        self.assertIn('25.0%', text)
        self.assertIn('10.0%', text)
        self.assertIn('2200 (2100-2300)', text)
        self.assertIn('130', text)
        self.assertIn('100%', text)


class BenchTests(unittest.TestCase):
    def test_bench_runs_every_setting_on_each_uploaded_clip(self):
        with tempfile.TemporaryDirectory() as d:
            sess = os.path.join(d, '2026-10-22-licht')
            os.makedirs(sess)
            entries = []
            for k, (ref, n) in enumerate((('Mach das Licht an.', 800),
                                          ('Wie warm ist es?', 1200)), 1):
                name = f'{k:03d}.wav'
                with wave.open(os.path.join(sess, name), 'wb') as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                    w.writeframes(tone(n, k))
                entries.append({'file': name, 'ref': ref, 'floor_dbfs': -84.0,
                                'session_lang': 'de'})
            with open(os.path.join(sess, 'index.jsonl'), 'w') as f:
                f.writelines(json.dumps(e) + '\n' for e in entries)

            fb = FakeBoard(texts={800: 'Mach das Licht an.', 1200: 'Wie warm ist es'})
            out = os.path.join(d, 'r.csv')
            argv = ['whistle_bench.py', 'port', sess, '--out', out, '--depth', '3,8',
                    '--lang', 'de;de,en']
            buf = io.StringIO()
            with patch.object(bench, 'Board', lambda port, reset: ws.Board(port, ser=fb)), \
                    patch.object(sys, 'argv', argv), redirect_stdout(buf):
                bench.main()
            with open(out, newline='') as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 8)
        self.assertEqual([(r[0], r[1]) for r in fb.runs][:4],
                         [(3, 'de'), (3, 'de,en'), (8, 'de'), (8, 'de,en')])
        self.assertTrue(all(r['word_errors'] == '0' for r in rows))
        self.assertIn('2 clips x 4 settings = 8 runs', buf.getvalue())

    def test_bench_stops_when_model_is_too_shallow(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'index.jsonl'), 'w'):
                pass
            with wave.open(os.path.join(d, '001.wav'), 'wb') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes(tone(10))
            with open(os.path.join(d, 'index.jsonl'), 'w') as f:
                f.write(json.dumps({'file': '001.wav', 'ref': 'x'}) + '\n')
            fb = FakeBoard(layers=4)
            with patch.object(bench, 'Board', lambda port, reset: ws.Board(port, ser=fb)), \
                    patch.object(sys, 'argv', ['b', 'port', d]):
                with self.assertRaises(SystemExit) as e:
                    bench.main()
        self.assertIn('4 decoder layers', str(e.exception))
        self.assertFalse(fb.runs)


if __name__ == '__main__':
    unittest.main()
