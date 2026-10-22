#!/usr/bin/env bash
# Check that every entry in an app's sdkconfig.defaults made it into the
# generated sdkconfig. ESP-IDF drops an unknown or unsatisfiable option with
# only a warning, so a renamed Kconfig symbol or a missing dependency would
# otherwise leave a build that silently ignores part of its configuration.
#
#   tools/check_sdkconfig.sh esp32/whistle_demo
#
# Run after `idf.py build` (or `reconfigure`) in that directory.
set -euo pipefail

app="${1:?usage: $0 APP_DIR}"
defaults="$app/sdkconfig.defaults"
config="$app/sdkconfig"
[ -f "$config" ] || { echo "$config missing; build first" >&2; exit 2; }

bad=0
while IFS= read -r line; do
  case "$line" in
    CONFIG_*=*|"# CONFIG_"*" is not set") ;;
    *) continue ;;
  esac
  # "CONFIG_X=n" is written out as "# CONFIG_X is not set".
  case "$line" in CONFIG_*=n) line="# ${line%=n} is not set" ;; esac
  if ! grep -qxF "$line" "$config"; then
    sym="${line#\# }"; sym="${sym%%[= ]*}"
    got="$(grep -E "^(# )?${sym}[= ]" "$config" || echo "(absent)")"
    echo "::error file=$defaults::$line -> $got"
    bad=1
  fi
done < "$defaults"

[ "$bad" -eq 0 ] && echo "$defaults: all entries applied"
exit "$bad"
