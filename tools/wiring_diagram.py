#!/usr/bin/env python3
"""Draw the wiring diagram: INMP441 + push button on an ESP32-S3 DevKitC-1-style board.

  python3 tools/wiring_diagram.py docs/wiring.svg
  rsvg-convert -z 2 docs/wiring.svg -o docs/wiring.png
"""
import sys

LEFT = ["3V3", "3V3", "RST", "GPIO4", "GPIO5", "GPIO6", "GPIO7", "GPIO15", "GPIO16", "GPIO17",
        "GPIO18", "GPIO8", "GPIO3", "GPIO46", "GPIO9", "GPIO10", "GPIO11", "GPIO12", "GPIO13",
        "GPIO14", "5V", "GND"]
RIGHT = ["GND", "TX", "RX", "GPIO1", "GPIO2", "GPIO42", "GPIO41", "GPIO40", "GPIO39", "GPIO38",
         "GPIO37", "GPIO36", "GPIO35", "GPIO0", "GPIO45", "GPIO48", "GPIO47", "GPIO21", "GPIO20",
         "GPIO19", "GND", "GND"]

C = dict(v3="#e03131", gnd="#343a40", sck="#e8590c", ws="#2f9e44", sd="#1971c2", btn="#9c36b5")
Y0, PITCH = 130, 28
BX0, BX1 = 500, 760          # board edges
LX, RX = 518, 742            # header pin columns
MX = 300                     # mic pin column (right edge of the INMP441)


def py(i):                   # 0-based header row -> y
    return Y0 + i * PITCH


used_left = {0: "v3", 3: "sck", 4: "ws", 5: "sd", 6: "btn", 21: "gnd"}
o = []
w = o.append
w('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 900" width="820" height="900" '
  'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">')
w('<rect width="820" height="900" rx="18" fill="#ffffff"/>')
w('<text x="40" y="52" font-size="24" font-weight="700" fill="#212529">Whistle on ESP32-S3: wiring</text>')
w('<text x="40" y="80" font-size="14" fill="#495057">INMP441 I²S microphone + one push button. '
  'The RGB LED is the one already on the board.</text>')

# --- board ---
w(f'<rect x="{BX0}" y="{Y0-40}" width="{BX1-BX0}" height="{py(21)-Y0+90}" rx="10" fill="#1e2a38"/>')
w(f'<rect x="588" y="{Y0-22}" width="84" height="150" rx="4" fill="#adb5bd"/>')
w(f'<rect x="596" y="{Y0-14}" width="68" height="134" rx="2" fill="#ced4da"/>')
w(f'<text x="630" y="{Y0+46}" font-size="12" font-weight="700" fill="#343a40" text-anchor="middle">ESP32-S3</text>')
w(f'<text x="630" y="{Y0+62}" font-size="11" fill="#343a40" text-anchor="middle">N16R8</text>')
w(f'<text x="630" y="{Y0+78}" font-size="9" fill="#495057" text-anchor="middle">16 MB flash</text>')
w(f'<text x="630" y="{Y0+90}" font-size="9" fill="#495057" text-anchor="middle">8 MB PSRAM</text>')
for i, name in enumerate(LEFT):
    hl = used_left.get(i)
    w(f'<circle cx="{LX}" cy="{py(i)}" r="6" fill="#d4a017" stroke="{C[hl] if hl else "#8a6d0b"}" '
      f'stroke-width="{3 if hl else 1}"/>')
    w(f'<text x="{LX+14}" y="{py(i)+4}" font-size="11" fill="{"#ffffff" if hl else "#8fa3b8"}" '
      f'font-weight="{700 if hl else 400}">{name}</text>')
for i, name in enumerate(RIGHT):
    hl = name == "GPIO48"
    w(f'<circle cx="{RX}" cy="{py(i)}" r="6" fill="#d4a017" stroke="#8a6d0b" stroke-width="1"/>')
    w(f'<text x="{RX-14}" y="{py(i)+4}" font-size="11" text-anchor="end" fill="{"#ffffff" if hl else "#8fa3b8"}" '
      f'font-weight="{700 if hl else 400}">{name}</text>')
# onboard LED next to GPIO48
ly = py(15)
w(f'<rect x="616" y="{ly-14}" width="28" height="28" rx="3" fill="#f8f9fa"/>')
w(f'<circle cx="630" cy="{ly}" r="8" fill="#ff6b6b"/><circle cx="627" cy="{ly-2}" r="3" fill="#ffe3e3"/>')
w(f'<path d="M646 {ly} H 680" stroke="#ffffff" stroke-width="1.5" stroke-dasharray="3 3"/>')
w(f'<text x="630" y="{ly+32}" font-size="10" fill="#ffffff" text-anchor="middle">WS2812 RGB LED</text>')
w(f'<text x="630" y="{ly+45}" font-size="10" fill="#8fa3b8" text-anchor="middle">(onboard, GPIO48)</text>')
# BOOT / RST buttons and USB ports
by = py(21) + 30
for x, t in ((560, "BOOT"), (676, "RST")):
    w(f'<rect x="{x}" y="{by-26}" width="24" height="16" rx="2" fill="#495057"/>'
      f'<text x="{x+12}" y="{by+4}" font-size="9" fill="#8fa3b8" text-anchor="middle">{t}</text>')
uy = Y0 - 40 + py(21) - Y0 + 90
for x, t, sub in ((548, "USB", "native: plug in here"), (652, "UART", "serial bridge")):
    w(f'<rect x="{x}" y="{uy-4}" width="60" height="26" rx="4" fill="#868e96" stroke="#495057"/>')
    w(f'<text x="{x+30}" y="{uy+44}" font-size="12" font-weight="700" fill="#212529" text-anchor="middle">{t}</text>')
    w(f'<text x="{x+30}" y="{uy+60}" font-size="10" fill="#495057" text-anchor="middle">{sub}</text>')

# --- INMP441 ---
mic = [("VDD", 0, "v3"), ("SCK", 3, "sck"), ("WS", 4, "ws"), ("SD", 5, "sd"), ("L/R", 7, "gnd"), ("GND", 8, "gnd")]
top, bot = py(0) - 34, py(8) + 34
w(f'<rect x="120" y="{top}" width="{MX-120}" height="{bot-top}" rx="10" fill="#1864ab"/>')
w(f'<circle cx="175" cy="{(top+bot)//2}" r="26" fill="#adb5bd"/><circle cx="175" cy="{(top+bot)//2}" r="7" fill="#212529"/>')
w(f'<text x="175" y="{bot-46}" font-size="15" font-weight="700" fill="#ffffff" text-anchor="middle">INMP441</text>')
w(f'<text x="175" y="{bot-30}" font-size="10" fill="#d0ebff" text-anchor="middle">I²S microphone</text>')
for label, row, col in mic:
    y = py(row)
    w(f'<circle cx="{MX-12}" cy="{y}" r="6" fill="#d4a017"/>')
    w(f'<text x="{MX-24}" y="{y+4}" font-size="12" font-weight="700" fill="#ffffff" text-anchor="end">{label}</text>')

# --- wires ---
def wire(d, col):
    w(f'<path d="{d}" fill="none" stroke="{C[col]}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')

for label, row, col in mic[:4]:
    wire(f"M{MX-12} {py(row)} H {LX-6}", col)
GX = 400                                         # ground bus
gy = py(21)
wire(f"M{MX-12} {py(7)} H {GX} M{MX-12} {py(8)} H {GX} M{GX} {py(7)} V {gy} H {LX-6}", "gnd")
w(f'<circle cx="{GX}" cy="{py(8)}" r="5" fill="{C["gnd"]}"/>')

# --- push button, between GPIO7 and the ground bus ---
BXc, bt, bb = 460, py(12), py(14)
wire(f"M{BXc} {bt} V {py(6)} H {LX-6}", "btn")
wire(f"M{BXc} {bb} V {gy}", "gnd")
w(f'<circle cx="{BXc}" cy="{gy}" r="5" fill="{C["gnd"]}"/>')
w(f'<rect x="{BXc-20}" y="{bt}" width="40" height="{bb-bt}" rx="4" fill="#212529"/>')
w(f'<circle cx="{BXc}" cy="{(bt+bb)//2}" r="12" fill="#495057" stroke="#868e96" stroke-width="2"/>')
w(f'<text x="{GX-14}" y="{(bt+bb)//2-4}" font-size="13" font-weight="700" fill="#212529" text-anchor="end">Push button</text>')
w(f'<text x="{GX-14}" y="{(bt+bb)//2+13}" font-size="11" fill="#495057" text-anchor="end">hold to talk</text>')

# wire labels near the mic
for label, row, col in mic:
    pass

# --- connection table ---
rows = [("v3", "INMP441 VDD", "3V3", "3.3 V only, not 5 V"),
        ("gnd", "INMP441 GND", "GND", ""),
        ("gnd", "INMP441 L/R", "GND", "selects the left channel"),
        ("sck", "INMP441 SCK", "GPIO4", "bit clock"),
        ("ws", "INMP441 WS", "GPIO5", "word select"),
        ("sd", "INMP441 SD", "GPIO6", "data"),
        ("btn", "Button", "GPIO7 + GND", "internal pull-up"),
        (None, "WS2812 LED", "GPIO48", "already on the board")]
ty = py(15) - 6
w(f'<text x="40" y="{ty}" font-size="14" font-weight="700" fill="#212529">Connections</text>')
for k, (col, a, b, note) in enumerate(rows):
    y = ty + 24 + k * 22
    if col:
        w(f'<rect x="40" y="{y-10}" width="18" height="6" rx="3" fill="{C[col]}"/>')
    w(f'<text x="68" y="{y-3}" font-size="12" fill="#212529">{a}</text>')
    w(f'<text x="168" y="{y-3}" font-size="12" font-weight="700" fill="#212529">→ {b}</text>')
    w(f'<text x="268" y="{y-3}" font-size="11" fill="#868e96">{note}</text>')

fy = 860
w(f'<text x="40" y="{fy}" font-size="11" fill="#868e96">Pin positions follow the ESP32-S3-DevKitC-1 header. '
  'Clone boards move pins around and may swap the two USB ports,</text>')
w(f'<text x="40" y="{fy+16}" font-size="11" fill="#868e96">so go by the labels printed on yours. '
  'DevKitC-1 v1.1 puts the LED on GPIO38 instead of GPIO48.</text>')
w('</svg>')
open(sys.argv[1], "w").write("\n".join(o) + "\n")
