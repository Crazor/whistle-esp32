"""The console protocol of esp32/whistle_stt, shared by whistle_collect.py and
whistle_bench.py.

    b = Board('/dev/cu.usbmodem...', reset=False)
    cfg = b.cfg()                      # {'depth': '8', 'layers': '0,1,...', ...}
    b.upload(samples, floor_dbfs)      # int16 samples; floor None: unknown
    res = b.run()                      # the RES line as a dict
"""
import array, base64, queue, re, shlex, sys, threading, time

BAUD = 921600
SAMPLE_RATE = 16000
UPLOAD_BYTES = 768        # per base64 line: 1024 characters, one OK each


def parse_kv(line):
    """'RES depth=4 text="a \\"b\\""' -> {'depth': '4', 'text': 'a "b"'}"""
    out = {}
    for tok in shlex.split(line.split(' ', 1)[1] if ' ' in line else '', posix=True):
        if '=' in tok:
            k, v = tok.split('=', 1)
            out[k] = v
    return out


class BoardError(RuntimeError):
    pass


class Board:
    """The board on a serial port. A reader thread queues every line; the
    callers below take the ones they wait for and `echo` gets the rest."""

    def __init__(self, port, reset=False, echo=None, ser=None):
        if ser is None:
            import serial
            # DTR and RTS low before opening: on the ESP32-P4-WIFI6's bridge
            # DTR pulls BOOT low (download mode at reset, and the record
            # button while running) and RTS holds the chip in reset.
            ser = serial.Serial()
            ser.port, ser.baudrate, ser.timeout = port, BAUD, 0.2
            ser.dtr = ser.rts = False
            ser.open()
            if reset:
                ser.rts = True
                time.sleep(0.15)
                ser.rts = False
        self.s = ser
        self.echo = echo or (lambda line: None)
        self.lines = queue.Queue()
        self._stop = False
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()
        if reset:
            self.wait(lambda l: l.startswith('CFG '), 30, 'the board to boot')

    def _read(self):
        buf = b''
        while not self._stop:
            try:
                chunk = self.s.read(4096) if self.s.in_waiting else self.s.read(1)
            except Exception:
                if self._stop:
                    return
                raise
            buf += chunk
            while b'\n' in buf:
                raw, buf = buf.split(b'\n', 1)
                line = raw.decode('utf-8', 'replace').rstrip('\r')
                if line:
                    self.lines.put(line)

    def close(self):
        self._stop = True
        self.s.close()

    def send(self, line):
        self.s.write((line + '\n').encode())

    def get(self, timeout):
        """The next line, or None after `timeout` seconds."""
        try:
            return self.lines.get(timeout=timeout)
        except queue.Empty:
            return None

    def wait(self, match, timeout, what):
        """Lines until `match(line)` is true; others go to `echo`. A reply
        starting with ERR raises."""
        end = time.time() + timeout
        while True:
            line = self.get(max(0.0, end - time.time()))
            if line is None:
                raise BoardError(f'no answer from the board (waiting for {what})')
            if match(line):
                return line
            if line.startswith('ERR '):
                raise BoardError(line[4:])
            self.echo(line)

    def command(self, line, prefix, timeout=5):
        self.send(line)
        return self.wait(lambda l: l.startswith(prefix), timeout, f'{prefix.strip()} after "{line}"')

    def cfg(self, **kw):
        """`set` with the given keys (none: just ask); returns the CFG line."""
        args = ''.join(f' {k}={v}' for k, v in kw.items())
        return parse_kv(self.command('set' + args, 'CFG '))

    def dump(self, on):
        return parse_kv(self.command('dump ' + ('on' if on else 'off'), 'CFG '))

    def upload(self, samples, floor_dbfs):
        """Send a clip (a sequence of int16) for `run`, a line per OK."""
        a = array.array('h', samples)
        if sys.byteorder == 'big':
            a.byteswap()
        data = a.tobytes()
        fl = 'none' if floor_dbfs is None else f'{floor_dbfs:.4f}'
        self.command(f'clip n={len(a)} floor_dbfs={fl}', 'OK')
        for i in range(0, len(data), UPLOAD_BYTES):
            self.command(base64.b64encode(data[i:i + UPLOAD_BYTES]).decode(), 'OK')
        self.command('clip end', 'OK clip')

    def run(self, timeout=120):
        """Transcribe the uploaded clip; the RES line as a dict."""
        return parse_kv(self.command('run', 'RES ', timeout))


class Capture:
    """Collects one push-to-talk clip from the lines `dump on` produces:
    feed() every line; it returns None until the clip is complete, then
    {'res': RES dict, 'pcm': s16le bytes, 'floor_dbfs': float or None}, or
    {'too_short': True} for a press too short to transcribe."""

    def __init__(self):
        self.res, self.pcm = None, bytearray()

    def feed(self, line):
        if line.startswith('PCM end'):
            fl = parse_kv(line).get('floor_dbfs', 'none')
            out = {'res': self.res or {}, 'pcm': bytes(self.pcm),
                   'floor_dbfs': None if fl == 'none' else float(fl)}
            self.__init__()
            return out
        if line.startswith('PCM '):
            self.pcm += base64.b64decode(line[4:])
        elif line.startswith('RES '):
            self.res = parse_kv(line)
        elif line.startswith('EVT too_short'):
            self.__init__()
            return {'too_short': True}
        elif line.startswith('EVT recording'):
            self.__init__()
        return None


_PUNCT = re.compile(r'[^\w\s]|_')


def normalise(text):
    """Lowercase, no punctuation, single spaces."""
    return ' '.join(_PUNCT.sub(' ', text.lower()).split())


def edits(ref, hyp):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]
