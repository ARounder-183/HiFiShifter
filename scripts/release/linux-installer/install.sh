#!/usr/bin/env bash
# HiFiShifter setup.
#
# [Why a three-line forwarder] The wizard needs four files (run.sh, ui.sh, paths.sh,
# strings.sh) that must stay together, and the uninstaller is the same wizard started with
# --uninstall. Keeping the entry point at the top of the extracted folder is what makes
# "./install.sh" -- the command every README shows -- keep working, while the implementation
# lives in installer/ with its siblings.
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$here/installer/run.sh" "$@"
