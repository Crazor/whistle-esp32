#!/usr/bin/env python3
"""Monitor esp32/mic_test and save every push-to-talk clip as a WAV.

    python tools/mic_capture.py /dev/ttyACM0 [secs] [--out DIR] [--baud N]

Prints the EVT/CLIP lines as they arrive and writes clip_NNN.wav (16 kHz
s16le; stereo when the firmware sends both I2S slots) for each PCM block. The
port is opened with DTR and RTS both low and left that way: the open-time
glitch pyserial makes by default can latch the chip into download mode, and
on the ESP32-P4-WIFI6's USB bridge DTR alone holds BOOT, the record button,
down. To catch the boot
lines, reset with `esptool.py --after watchdog_reset read_mac` first.
"""
import argparse, base64, os, re, sys, time, wave
import serial

ap = argparse.ArgumentParser()
ap.add_argument('port')
ap.add_argument('secs', nargs='?', type=float, default=120)
ap.add_argument('--out', default='clips')
ap.add_argument('--baud', type=int, default=921600)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
# Continue numbering after the highest existing clip_NNN.wav.
idx = 1 + max([int(m.group(1)) for f in os.listdir(a.out)
               if (m := re.fullmatch(r'clip_(\d{3})\.wav', f))], default=-1)
# Open with both lines low. pyserial raises DTR and RTS on open by default,
# which can latch the chip into download mode or hold BOOT down.
s = serial.Serial()
s.port, s.baudrate, s.timeout, s.dtr, s.rts = a.port, a.baud, 0.3, False, False
s.open()
pcm, bad, t0 = bytearray(), 0, time.time()
while time.time() - t0 < a.secs:
    line = s.readline().decode('utf-8', 'replace').rstrip()
    if not line:
        continue
    if line.startswith('PCM end'):
        kv = dict(t.split('=') for t in line.split()[2:] if '=' in t)
        rate, ch = int(kv.get('rate', 16000)), int(kv.get('ch', 1))
        path = os.path.join(a.out, f'clip_{idx:03d}.wav')
        with wave.open(path, 'wb') as w:
            w.setnchannels(ch); w.setsampwidth(2); w.setframerate(rate)
            w.writeframes(bytes(pcm))
        print(f'  saved {path} ({len(pcm) // (2 * ch)} frames, {ch} ch)'
              + (f'  WARNING {bad} damaged lines zero-filled' if bad else ''),
              flush=True)
        idx, pcm, bad = idx + 1, bytearray(), 0
    elif line.startswith('PCM '):
        try:
            pcm += base64.b64decode(line[4:], validate=True)
        except ValueError:
            # A damaged line: keep the timing by filling its 96 bytes with
            # silence, and say so rather than saving a quietly wrong clip.
            pcm += bytes(96)
            bad += 1
    else:
        print(line, flush=True)
s.close()
