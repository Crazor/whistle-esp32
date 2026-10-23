# Whistle speech recognition on an ESP32-P4

Hold a button and say *"flash the light blue"*. The LED flashes blue.

The speech recognition runs entirely on a microcontroller: no network, no
cloud, no phone. It transcribes whatever you say rather than matching a fixed
list of phrases, and it understands commands in seven languages.

The model is [Whistle](https://huggingface.co/Cactus-Compute/whistle) by
Cactus Compute (Apache 2.0). Cactus ships no ESP32 build and no source for
Whistle's engine. Everything here is an independent C99 implementation with no
dependencies, checked tensor by tensor against the official engine.

## What you need

| Part | Notes |
|---|---|
| [Waveshare ESP32-P4-WIFI6](https://www.waveshare.com/esp32-p4-wifi6.htm) | ESP32-P4NRW32: 32 MB PSRAM in the package, 32 MB flash. The model alone is 14.5 MB. |
| WS2812 RGB LED | a single-pixel breakout; the board has no RGB LED of its own |
| USB-C cable | to the board's Type-C port (the USB-UART bridge) |

The microphone (an onboard mic behind the ES8311 codec) and the push-to-talk
button (**BOOT**) are already on the board.

This port started from an ESP32-S3 build; `whistle_demo`, `whistle_stt` and
`mic_test` target the P4. `needle_demo`, `led_probe` and `psram_test` are still the S3
versions.

## Wiring

Only the LED needs wiring:

| From | To | |
|---|---|---|
| WS2812 VCC | 3V3 or 5V | per your breakout; 5 V parts usually take a 3.3 V data line at this length |
| WS2812 GND | GND | |
| WS2812 DIN | GPIO23 | `LED_GPIO` in `esp32/components/board/include/board.h` |

The 40-pin header follows the Raspberry Pi Pico pinout, but Waveshare documents
it only as a picture. **Check the silkscreen** for where GPIO23 is, or change
`LED_GPIO`. `board.h` lists the pins the board already uses.

| Onboard | Pins |
|---|---|
| ES8311 I²S | MCLK GPIO13, BCLK GPIO12, WS GPIO10, ADC data GPIO11 |
| ES8311 I²C | SDA GPIO7, SCL GPIO8, address 0x18 |
| BOOT button | GPIO35 |

**Check the microphone before anything else.** `esp32/mic_test` records while
BOOT is held, and `tools/mic_capture.py` saves each clip as a WAV you can
listen to. Speech should sit well above the noise floor, with peaks well under
0 dBFS. Set the codec gain with `BOARD_MIC_GAIN_6DB` in `board.h`.

## Setup

You need [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/get-started/)
v5.5 or newer, Python 3 and CMake.

**Check the chip revision first.** The build targets revision 3.x, as current
boards carry (v3.2 here). Earlier boards shipped with v1.x chips (v1.3), and
the two are not binary compatible:

```bash
esptool.py --chip esp32p4 -p /dev/ttyACM0 chip_id    # "Chip is ESP32-P4 (revision v3.2)"
```

On a v1.x chip, add `CONFIG_ESP32P4_SELECTS_REV_LESS_V3=y` and
`CONFIG_ESP32P4_REV_MIN_100=y` to `sdkconfig.defaults` in `esp32/whistle_demo`,
`esp32/whistle_stt` and `esp32/mic_test`. Also replace `ESP_DEFAULT_CPU_FREQ_MHZ_400` with
`ESP_DEFAULT_CPU_FREQ_MHZ_360`, since 1.x tops out there. Then delete
`sdkconfig` and `build` and rebuild.

On Debian or Ubuntu, including a fresh cloud container, `./setup-linux.sh`
installs all of it: system packages, `.venv` from `requirements.txt`, and
ESP-IDF v5.5.6 with the ESP32-P4 and ESP32-S3 toolchains (override with
`IDF_VERSION` / `IDF_TARGETS`). It is idempotent, so it works as a
session-start script, and puts `idf.py` on the PATH of later shells.

```bash
# 1. host tools (setup-linux.sh does this)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. fetch Whistle (16.9 MB) and cut it down to 4 decoder layers (14.5 MB)
mkdir -p model/whistle && .venv/bin/python -c "
from huggingface_hub import hf_hub_download; import shutil
shutil.copy(hf_hub_download('Cactus-Compute/whistle','whistle.cact'), 'model/whistle/whistle.cact')"
.venv/bin/python tools/whistle_strip.py model/whistle/whistle.cact 4 model/whistle/whistle_d4.cact

# 3. firmware (ESP-IDF must be on PATH)
. $HOME/esp/esp-idf/export.sh
cd esp32/whistle_demo
idf.py set-target esp32p4 && idf.py build
idf.py -p /dev/ttyACM0 flash                  # bootloader, partition table, app
cd ../..

# 4. the model, into its own flash partition (~100 s)
esptool.py --chip esp32p4 -p /dev/ttyACM0 -b 921600 \
    write_flash 0x210000 model/whistle/whistle_d4.cact
```

Find the port with `ls /dev/cu.usb*` (macOS) or `ls /dev/ttyACM*` (Linux).
The console runs at 921600 baud; `idf.py monitor` picks that up from the
build, and the tools here default to it.
The model only needs flashing once. After a code change, repeat step 3.

**Why cut the model down?** The full model is bigger than the 16 MB flash of
the S3 board this started on. Whistle's decoder is laddered: every depth from
2 layers up was trained to work on its own. `whistle_strip.py` keeps only the
layers a depth uses. The P4 board's 32 MB flash also holds the full model, but
the accuracy figures below were measured and tuned at depth 4.

At boot the firmware copies the model from flash into PSRAM (`EVT
model_in_psram`). PSRAM streams the weights far faster than the flash
mapping.

### Watching it work

```bash
.venv/bin/python tools/needle_tui.py --whistle --serial /dev/ttyACM0
```

The console shows each transcript, the detected language, the LED action and
the time each phase took. The board works without it, too.

## Using it

The LED glows dim blue for about a second while the model loads, then goes
off. That means it is ready.

1. **Hold** BOOT. The LED turns **white** while it records.
2. **Say** a command.
3. **Release.** The LED **pulses dimly** while it transcribes, then does what
   you said.

A command names a colour (red, green, blue, yellow, purple or white) and can
add **flash** or **blink**, a blink rate and a duration. The default duration
is 3 seconds.

| Say | LED |
|---|---|
| "Make it red." | solid red |
| "Turn the light green for five seconds." | green for 5 s |
| "Flash the light blue." | flashing blue |
| "Blink purple every half a second." | purple, blinking twice a second |

**Two red blinks** mean it heard only noise. **One amber blink** means it
heard words but no colour.

Short phrases work better than a bare word: "make it red" was recognised far
more often than "red" alone (86-93% vs 44% across six test voices).

### Other languages

Language detection is automatic. Every language below is on the board at the
same time.

| Language | Example | Colours |
|---|---|---|
| English | *Flash the light blue.* | red, green, blue, yellow, purple, white |
| French | *Fais clignoter la lumière en violet.* | rouge, vert, bleu, jaune, violet, blanc |
| Spanish | *Pon la luz amarilla.* | rojo, verde, azul, amarillo, morado, blanco |
| German | *Lass das Licht rot blinken.* | rot, grün, blau, gelb, lila, weiß |
| Italian | *Fai lampeggiare la luce verde.* | rosso, verde, blu, giallo, viola, bianco |
| Dutch | *Laat het licht blauw knipperen.* | rood, groen, blauw, geel, paars, wit |
| Polish | *Niech światło miga na czerwono.* | czerwony, zielony, niebieski, żółty, fioletowy, biały |

The blink words are *clignoter*, *parpadear*, *blinken*, *lampeggiare*,
*knipperen* and *migać*. Numbers from one to ten and "half" work in each
language, as in *toutes les deux secondes* or *alle zwei Sekunden*.

## Speech to text: whistle_stt

`esp32/whistle_stt` is a plain transcriber: hold BOOT, speak, release, and the
text appears on the console. It has no LED and no colour commands, no keyword
bias and no retries in other languages. It is the front end of a local voice
assistant, and the bench its transcription is measured on. It uses
`whistle_demo`'s partition table, so the model already in flash serves both;
build and flash it as in step 3 of the setup, in `esp32/whistle_stt`. With
the full `whistle.cact` flashed (31.4 MB partition, about 4.5 minutes to
write), every depth from 2 to 8 is available.

The console prints `EVT` status lines (`booting`, `ready`, `recording`,
`transcribing`, `error ...`), the transcript as `TXT`, and then one `RES`
line for tools:

```
RES depth=<n> lang=<setting> lang_detected=<l> audio_ms= speech_ms= encode_ms= decode_ms= total_ms= tokens= text="..."
```

Commands, one per line on the console, take effect between clips:

| Command | Does |
|---|---|
| `set depth=<n>` | decoder layers, 2 up to the flashed model's (default: all). Switching is instant |
| `set lang=<l>` | `auto` detects among all seven languages (default); one language (`de`) skips detection; a list (`de,en`) detects among those only |
| `dump on` / `dump off` | after each live transcript, send the clip as transcribed (press and release clicks cut off) as `PCM` base64 lines of s16le, then `PCM end ... n=<samples> floor_dbfs=<x>` |
| `clip n=<samples> floor_dbfs=<x>` | upload a clip of at most 8 s: base64 s16le lines, each answered `OK` before the next, then `clip end`. `floor_dbfs=none` estimates the floor from the clip |
| `run` | transcribe the uploaded clip with the current settings, gated against the uploaded floor as when it was recorded |

`set depth=4 lang=de` sets both, and `set` alone only shows the settings.
Each answers with a line like `CFG depth=4 layers=0,3,5,7 lang=de
model_layers=8 dump=off`, or `ERR ...`. Nothing is stored in flash: a restart
goes back to full depth and `lang=auto`.

### Recording and replaying phrases

`tools/whistle_collect.py` records a session of phrases through the board.
Put one prompt per line in `clips/<date>-<topic>/phrases.txt`, with `#`
comments and a `# lang: de` line for the session's language:

```bash
.venv/bin/python tools/whistle_collect.py /dev/cu.usbmodem... clips/2026-10-22-licht
```

It shows each prompt; hold BOOT, say it, release. The board's transcript
appears, Enter keeps the take and `r` drops it (so does pressing BOOT again).
`s` skips a prompt, `f` records a free phrase and asks what you said, `q`
quits. Other lines, such as `set depth=4`, go to the board. Each take is saved
as `NNN.wav` (16 kHz mono s16le) with a line in `index.jsonl`: the reference
text, the board's transcript, the noise floor, the depth and language at
capture, and the time. Prompts already in the index are skipped, so a session
can be continued. `clips/` holds recordings of your voice and is not
committed.

`tools/whistle_bench.py` replays sessions on the board:

```bash
.venv/bin/python tools/whistle_bench.py /dev/cu.usbmodem... clips/2026-10-22-licht \
    --depth 3,4,8 --lang de,auto --out results.csv
```

The defaults are depths 3, 4 and 8, each with `lang=de` and `lang=auto`. It
prints the number of runs and an estimate first (about 4 s a run), uploads
each clip once and runs it under every setting, and writes a CSV row per run.
A setting that is itself a list needs `;` between settings:
`--lang "de;de,en;auto"`. The summary per setting has the word and character
error rates (both texts lowercased, punctuation removed; CER is fairer to
German compounds, since "Braut Kleid" for "Brautkleid" is two wrong words but
one wrong character), the median and interquartile range of the total time,
the encoding time per second of speech, the decoding time per token, and how
often the detected language was the session's. `--summary results.csv`
prints the summary of an earlier run again. Depths above 4 need the full
model; the tool checks `model_layers` first.

Both tools open the port with DTR and RTS low, as `tools/mon.py` does: on
this board DTR holds BOOT down and RTS resets the chip. `--reset` restarts
the firmware first.

## Performance

The P4 transcribes a 2-second command in about 3.5 s, 1.8 times as fast as
the S3. It runs at 400 MHz (revision 3.x) with the model in 200 MHz hex PSRAM;
the S3 runs at 240 MHz with the model in QIO flash. Depth 4, for a 2-second
English command, from release to result, as the firmware prints it
(`EVT prof ...`):

| Phase | P4 | S3 |
|---|---|---|
| log-mel spectrogram | 0.06 s | 0.1 s |
| conv stem | 0.34 s | 0.9 s |
| encoder (8 layers) | 1.57 s | 3.2 s |
| decoder | 1.47 s (11 tokens) | 2.2 s (9 tokens) |
| **total** | **3.5 s** | **6.3 s** |

On the P4 the total also includes 0.06 s to precompute the decoder's
cross-attention keys and values (`cross_kv`). Copying the model into PSRAM at
boot takes 1.3 s.

Across seven English and German commands of 1.2 to 3.0 seconds, the P4 took
2.5 to 4.8 s from release to result:

- **Encoding** (log-mel, conv stem and encoder) takes 0.9 s per second of
  speech, because the encoder sees every 80 ms frame. The longer you speak,
  the longer it takes.
- **Decoding** takes 122 to 137 ms per token, against about 240 ms on the S3.
  A command is 9 to 16 tokens.

Other languages can use more tokens: a 3.2-second French sentence takes
12.6 s on the S3. A transcript that names no colour costs more, because the
firmware decodes it again in the next two detected languages. On the P4 that
puts decoding at 300 to 760 ms per token: a 0.3-second word took 2.6 s, and a
2.8-second English sentence with no colour took 8.4 s.

The matmuls run in **int8 on the PIE vector unit**, 16 multiply-adds per
instruction. PIE is Espressif's SIMD extension: 128-bit registers holding
16 int8 values. The S3 has an Xtensa version (`ee.*` instructions) and the
P4 a RISC-V one (`esp.*`), so `nd_quant.c` has a kernel for each. The
firmware checks the kernel against plain C at boot
(`ERR int8_kernel_mismatch` if they disagree). Whistle's weights decode exactly to int8 codebook levels and
activations get one scale per 128-wide group, so accuracy does not change
(80% of colour commands right in int8 vs 78% in float, on 360 test clips).
The first working build took 23.9 s for the same command.

## Accuracy

Command accuracy means the colour and flash/solid are both right. It was
measured on synthetic voices (macOS voices, five phrasings, six colours), both
clean and mixed with real INMP441 noise, on the S3 build with its external
INMP441 microphone. The ES8311 path on the P4 board has not been scored yet.
The second row scores Cactus's own
full-size model, run through the official engine and the same parser:

| | en | de | fr | es | it | nl | pl |
|---|---|---|---|---|---|---|---|
| this board (depth 4), clean / noisy | 78 / 76% | 57 / 55% | 72 / 67% | 80 / 85% | 63 / 65% | 42 / 50% | 70 / 43% |
| official engine, full model | - | 37 / 32% | 63 / 48% | 63 / 63% | 42 / 40% | 13 / 17% | 40 / 30% |
| language detected correctly | 98% | 70% | 88% | 84% | 71% | 52% | 83% |

The board beats the full model mostly because the firmware biases decoding towards
each language's colour and blink words. When the first transcript names no
colour, it also decodes as the next two most likely languages. It accepts one
of those only if it names a colour and the model finds it nearly as probable,
within 0.2 nats per token. Looser rules turned ordinary sentences into
nonsense containing colour words.

English, Spanish and French are solid. German, Italian and Polish work. Dutch
is weak, and that comes from Whistle itself: its own detection picks Dutch for
only about half of these clips. Of 137 sentences with no colour in them, two
produced a command.

On general speech, depth 4 scores 11.7% word error rate on 73 LibriSpeech
utterances (depth 2: 15.3%). `tools/eval_whistle.py` and
`tools/eval_languages.py` reproduce these numbers given test clips.

## Troubleshooting

Most of these fail **silently**. The board keeps running and the model keeps
producing plausible text.

**No serial output, or garbage.** The console is UART0 through the board's
USB bridge at **921600** baud. The ROM bootloader's first lines are at 115200
and show as garbage at 921600, which is harmless.

**It records by itself, or never stops recording.** On this board's USB
bridge DTR is wired to BOOT, the push-to-talk button. A terminal that asserts
DTR holds the button down. The tools here keep DTR and RTS low. Configure
other terminals the same way.

**The bootloader refuses the image, or the app crashes at once.** The build
was made for the wrong chip revision. See Setup.

**ESP_ERROR_CHECK failed in `board.c` at boot.** The ES8311 did not answer
on I²C. The codec gets its clock from MCLK, so it is configured after the I²S
channel starts.

**"Thank you." from an empty room.** Whistle, the official engine included,
transcribes ten seconds of microphone noise as "Thank you." The firmware
measures the noise floor between presses and only runs the model when it
hears speech above it.

**Don't band-limit or boost the audio.** The Whistle blog says the front end
is band-limited to 250-3500 Hz, but the shipped filterbank spans 31-7969 Hz.
Band-passing a clip turned "turn the light red" into "turn the leg red".
Clipping hurts more than low volume, because the engine normalises gain
anyway.

**A click lowers the gain of everything else.** The engine scales each clip to
a peak of 1. A single spike, such as a button click, shrinks the speech. The
firmware keeps the mic clock running and drops the first few ms after the
button closes.

**No `double` on the board.** The S3's and P4's FPUs are single precision, so `double`
is emulated in software. One double accumulator and a double FFT twiddle made
a 2-second clip take 24 s. The output was still correct, so nothing on the
host showed the problem.

## Testing without the board

The host build runs the same engine on a computer:

```bash
cmake -S host -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
./build/wh_run model/whistle/whistle_d4.cact clip.wav --led-keywords --gate 12
.venv/bin/python tools/needle_tui.py --whistle --local clip.wav
```

`clip.wav` must be 16 kHz mono 16-bit. Add `-DCMAKE_C_FLAGS=-DND_PROFILE` to
the first line for per-phase timings.

The console's regression tests use the `rich` dependency installed above;
they need no model, board or serial connection. Run them from the repository
root:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### Verifying the engine

Whistle has no published source; the official engine is a prebuilt library.
`tools/whistle_ref.py` is a NumPy reference whose undocumented details were
worked out by probing that library (`tools/whistle_probe.py`), and every
tensor was matched by value against the training checkpoint.

```bash
.venv/bin/pip install cactus-needle safetensors
.venv/bin/python tools/whistle_cact.py model/whistle/whistle.cact      # tensors, sizes, depths
.venv/bin/python tools/check_whistle.py model/whistle/whistle.cact ./build/wh_run path/to/*.wav
```

| Check | Result |
|---|---|
| 676 tensors vs `whistle.safetensors` | 0 problems |
| C vs NumPy logits (incremental vs full-sequence) | < 6e-5, identical argmax |
| NumPy encoder vs the official engine, speech frames | median ~1%, worst < 3% |
| Transcripts vs `needle.transcribe()` | identical, or case/punctuation only |
| `--depth 2..4` vs the official `needle --audio-depth` | identical |
| `whistle_strip.py` output vs `--depth` on the full model | bit-identical logits |

`cactus-needle` sends anonymous usage counts by default. Set
`NEEDLE_TELEMETRY=0`; the tools here do.

## Layout

```
engine/              portable C99 engine, no dependencies
  nd_whistle.c         Whistle: log-mel, conv stem, encoder, decoder
  nd_ledcmd.c          transcript -> LED command, in seven languages
  nd_quant.c           Cactus-Quants kernels, including the int8 PIE path
  nd_cact.c            .cact v3/v4 container reader
  nd_model.c, ...      Needle 2 (see below)
esp32/whistle_demo/  push-to-talk speech -> LED firmware (ESP32-P4)
esp32/whistle_stt/   push-to-talk speech -> text, and the benchmark (ESP32-P4)
esp32/mic_test/      microphone and button check (ESP32-P4)
esp32/components/board/  ESP32-P4-WIFI6 pins and its ES8311 microphone
esp32/needle_demo/   the Needle 2 demo; also holds the engine's ESP-IDF component
host/                wh_run and other host tools
tools/               console UI, model tools, references, evaluation
docs/                wiring diagram (tools/wiring_diagram.py draws it)
```

## Built on Needle 2

This started as a port of Cactus's [Needle 2](https://github.com/andrisgauracs/needle-2-esp32),
a 45M-parameter tool-calling language model, to the same board. Whistle ships
in the newer Needle 3 container (`.cact` v4) and its decoder is a Needle 3
block, so the engine reads both formats. The Needle 2 demo is still here in
`esp32/needle_demo`; that repo's README explains it.

## Credits

Whistle, Needle 2 and their weights are by
[Cactus Compute](https://github.com/cactus-compute/needle), Apache 2.0. The
weights are not included here: step 2 of the setup downloads them from
Hugging Face. This code is Apache 2.0 too.
