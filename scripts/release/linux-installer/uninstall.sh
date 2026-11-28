#!/bin/sh
# HiFiShifter uninstaller.
#
# [Why this file is installed next to the app] The download is deleted long before the user
# wants the product gone; the macOS pkg solves the same problem by carrying its own uninstaller
# in the payload directory. This is that copy: it forwards to the wizard in installer/, so
# "remove what we installed" is the same code that installed it.
here="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "$here/installer/run.sh" --uninstall "$@"
