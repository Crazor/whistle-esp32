#!/usr/bin/env python3
"""Read the board's serial console (UART0 through the USB bridge, 921600 baud).

    python tools/mon.py /dev/ttyACM0 [secs] [--reset] [--send LINE ...]

DTR and RTS stay low. On the ESP32-P4-WIFI6 the bridge's DTR pulls BOOT low,
which the Whistle firmware reads as its button, and RTS holds the chip in
reset. --reset pulses RTS once to restart the firmware. Each --send LINE goes
to the console once the port is open (after a --reset, once the firmware is
up), e.g. --send 'join ssid="My Net" pass=secret' for esp32/wifi_test.
"""
import sys, time, serial

args = sys.argv[1:]
sends = []
while '--send' in args:
    i = args.index('--send')
    sends.append(args[i + 1])
    del args[i:i + 2]
port = args[0] if args else '/dev/ttyACM0'
secs = float(args[1]) if len(args) > 1 and args[1][0] != '-' else 20
reset = '--reset' in args

s = serial.Serial()
s.port, s.baudrate, s.timeout, s.dtr, s.rts = port, 921600, 0.3, False, False
s.open()
if reset:
    s.setRTS(True); time.sleep(0.15)
    s.setRTS(False)
    if sends:
        time.sleep(3)        # boot, and the C6 bring-up for wifi_test
for line in sends:
    s.write(line.encode() + b'\n')
t0 = time.time()
while time.time() - t0 < secs:
    line = s.readline()
    if line:
        sys.stdout.write(line.decode('utf-8', 'replace'))
        sys.stdout.flush()
s.close()
