# SVOX Pico

The text to speech engine from Android, by SVOX AG, under the Apache License
2.0 (`NOTICE`). Only the engine library (`pico/lib`) is here; the lingware
it reads is fetched by `tools/pico_lang.py`, and `tts/` wraps the API.

## Where it comes from

The library sources are those of
[naggety/picotts](https://github.com/naggety/picotts) at commit
`21089d223e177ba3cb7e385db8613a093dff74b5`. That is AOSP's
`platform/external/svox` with Debian's patches applied. The patches that touch
the library fix undefined behaviour: left shifts of negative numbers, 32-bit
`int` overflow in byte assembly, an out-of-bounds read in `picoacph.c` and an
off-by-one in `picosig2.c`. Debian's `volume` patch also halves the output
level (`picosig.c`, `vMod = 0.5f`), which keeps loud passages from clipping.

## Changed here

* `picopltf.h` takes the byte order from the compiler's `__BYTE_ORDER__`
  rather than `<endian.h>`, which only glibc has. With that change it builds
  on macOS and with ESP-IDF's newlib.

Nothing else is changed. Builds turn its warnings off (`-w`). It is 2009 code,
and its warnings are not ours to fix.

## Notes for the ESP32

* Output is identical between a 32-bit and a 64-bit host build that both use
  SSE floats (x87 floats differ in the last bits), so pointer width is not an
  issue.
* A few spots use `double`: the fast `exp` approximation per spectral bin in
  `picosig2.c`, plus a `sqrt` and an `exp` per frame. The P4 and S3 emulate
  `double` in software. By estimate that costs a few percent of a 400 MHz
  core. It is nowhere near the slowdown described under "No `double` on the
  board" in the top-level README.
* It allocates nothing itself. Everything, the lingware included, comes from
  the one memory area passed to `pico_initialize`.
