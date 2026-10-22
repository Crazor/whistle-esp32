#!/usr/bin/env bash
# Dependency installation. Nothing else.
#
# This installs the host toolchain, the Python environment for tools/ and
# tests/, and ESP-IDF. It does not fetch the model, configure, build or flash
# anything — that is the job of whatever runs after it. Written to be used as
# a session-start script for a fresh cloud container, which means:
#
#   * idempotent: every step checks before it acts, so a cached re-run is fast
#     and a partial previous run is repaired rather than duplicated
#   * quiet on success, loud on failure
#   * no "what to do next" output, because nobody is reading it
#
# The default target set is the ESP32-P4 being ported to plus the ESP32-S3 the
# firmware came from, so both stay buildable during the port. That pulls in
# the RISC-V and the Xtensa toolchain. IDF_TARGETS=all installs everything.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The 5.5 line is the README's minimum and carries the ESP32-P4 v3 silicon
# support; 6.x is a separate migration.
IDF_VERSION="${IDF_VERSION:-v5.5.6}"
IDF_PATH="${IDF_PATH:-$HOME/esp/esp-idf}"
IDF_TOOLS_PATH="${IDF_TOOLS_PATH:-$HOME/.espressif}"
# "all", or a comma-separated list such as "esp32s3,esp32p4".
IDF_TARGETS="${IDF_TARGETS:-esp32p4,esp32s3}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn\033[0m %s\n' "$*" >&2; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
fi

# --- system packages --------------------------------------------------------
# First group: the host build of engine/ (host/CMakeLists.txt) and the Python
# tools. Second: what ESP-IDF's installer and flashing need.
APT_PKGS=(
  build-essential cmake ninja-build git
  python3 python3-pip python3-venv
  wget flex bison gperf ccache dfu-util
  libusb-1.0-0 libffi-dev libssl-dev
  shellcheck
)

if command -v apt-get >/dev/null 2>&1; then
  missing=()
  for p in "${APT_PKGS[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    if [ -z "$SUDO" ] && [ "$(id -u)" -ne 0 ]; then
      warn "cannot install without root: ${missing[*]}"
    else
      log "apt: installing ${missing[*]}"
      $SUDO apt-get update -qq
      DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${missing[@]}"
    fi
  fi
else
  warn "no apt-get; install manually: ${APT_PKGS[*]}"
fi

# --- python environment -----------------------------------------------------
# Stamped on the contents of the requirement set, so a cached re-run costs a
# hash instead of a round-trip to the package index.
VENV="$REPO/.venv"
REQ="$REPO/requirements.txt"

[ -d "$VENV" ] || { log "creating $VENV"; python3 -m venv "$VENV"; }

req_fingerprint() {
  { cat "$REQ"; "$VENV/bin/python" --version; } | sha256sum | cut -c1-32
}
PY_STAMP="$VENV/.deps-$(req_fingerprint)"

if [ ! -f "$PY_STAMP" ]; then
  log "pip: installing host tool dependencies"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r "$REQ"
  rm -f "$VENV"/.deps-*        # drop stamps for superseded requirement sets
  touch "$PY_STAMP"
fi

# --- ESP-IDF ----------------------------------------------------------------
if [ -d "$IDF_PATH/.git" ]; then
  current="$(git -C "$IDF_PATH" describe --tags --exact-match 2>/dev/null || echo unknown)"
  if [ "$current" != "$IDF_VERSION" ]; then
    warn "ESP-IDF at $IDF_PATH is $current, wanted $IDF_VERSION; leaving it alone"
  fi
else
  log "cloning ESP-IDF $IDF_VERSION into $IDF_PATH"
  mkdir -p "$(dirname "$IDF_PATH")"
  # Clone into a temporary name and move it into place only when complete, so
  # an interrupted clone is retried next time instead of being mistaken for
  # an install by the check above.
  rm -rf "$IDF_PATH.partial"
  git clone --quiet --depth 1 -b "$IDF_VERSION" \
      --recursive --shallow-submodules \
      https://github.com/espressif/esp-idf.git "$IDF_PATH.partial"
  mv "$IDF_PATH.partial" "$IDF_PATH"
fi

# install.sh is idempotent but downloads and verifies a lot, so a cached re-run
# should skip it. A stamp keyed on version AND target set is more reliable than
# probing for toolchain directories, whose names differ per architecture.
STAMP="$IDF_TOOLS_PATH/.setup-linux-stamp-${IDF_VERSION}-${IDF_TARGETS//,/_}"
if [ ! -f "$STAMP" ]; then
  log "installing ESP-IDF tools for targets: $IDF_TARGETS"
  export IDF_TOOLS_PATH
  (cd "$IDF_PATH" && ./install.sh "$IDF_TARGETS")
  mkdir -p "$IDF_TOOLS_PATH"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$STAMP"
fi

# Make idf.py available to later shells. Guarded by a marker so repeated runs do
# not stack up, and tolerant of export.sh failing so a broken IDF cannot make
# every shell unusable.
BASHRC="$HOME/.bashrc"
MARKER="# >>> esp-idf >>>"
touch "$BASHRC"
if ! grep -qF "$MARKER" "$BASHRC"; then
  log "adding ESP-IDF to $BASHRC"
  cat >> "$BASHRC" <<RC

$MARKER
export IDF_PATH="$IDF_PATH"
export IDF_TOOLS_PATH="$IDF_TOOLS_PATH"
if [ -f "\$IDF_PATH/export.sh" ]; then
    . "\$IDF_PATH/export.sh" >/dev/null 2>&1 || true
fi
# <<< esp-idf <<<
RC
fi

# ~/.bashrc is only sourced for interactive shells, so the block above never
# runs for `bash -lc '...'`, which is how tool invocations and CI steps usually
# run. /etc/profile.d/*.sh is sourced for every login shell, so put the same
# export there too.
PROFILED="/etc/profile.d/esp-idf.sh"
if [ -d /etc/profile.d ] && { [ -w /etc/profile.d ] || [ -n "$SUDO" ]; }; then
  if [ ! -f "$PROFILED" ] || ! grep -qF "IDF_PATH=\"$IDF_PATH\"" "$PROFILED"; then
    log "adding ESP-IDF to $PROFILED"
    $SUDO tee "$PROFILED" >/dev/null <<RC
export IDF_PATH="$IDF_PATH"
export IDF_TOOLS_PATH="$IDF_TOOLS_PATH"
if [ -f "\$IDF_PATH/export.sh" ]; then
    . "\$IDF_PATH/export.sh" >/dev/null 2>&1 || true
fi
RC
  fi
else
  warn "cannot write $PROFILED (no root/sudo); idf.py may be unavailable in non-interactive login shells"
fi

log "dependencies ready"
