#!/usr/bin/env python3
"""Read the board's serial console (UART0 through the USB bridge, 921600 baud).

    python tools/mon.py /dev/ttyACM0 [secs] [--reset]

DTR and RTS stay low. On the ESP32-P4-WIFI6 the bridge's DTR pulls BOOT low,
which the Whistle firmware reads as its button, and RTS holds the chip in
reset. --reset pulses RTS once to restart the firmware.
"""
import sys, time, serial

port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
secs = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2][0] != '-' else 20
reset = '--reset' in sys.argv

s = serial.Serial()
s.port, s.baudrate, s.timeout, s.dtr, s.rts = port, 921600, 0.3, False, False
s.open()
if reset:
    s.setRTS(True); time.sleep(0.15)
    s.setRTS(False)
t0 = time.time()
while time.time() - t0 < secs:
    line = s.readline()
    if line:
        sys.stdout.write(line.decode('utf-8', 'replace'))
        sys.stdout.flush()
s.close()
