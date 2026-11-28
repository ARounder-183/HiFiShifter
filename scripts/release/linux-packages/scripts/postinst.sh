#!/bin/sh
# Post-install hook shared by the deb and the rpm (nfpm maps it to %post and to postinst).
#
# [Why every command is guarded] These caches are a convenience: a minimal container has neither
# update-desktop-database nor a running icon theme, and a package whose post-install script
# returns non-zero is left "half-configured" by dpkg -- the user then cannot remove it cleanly.
# Nothing here may fail the installation, and nothing here may touch user data.
set -eu

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q /usr/share/applications || true
fi
if command -v update-mime-database >/dev/null 2>&1; then
    update-mime-database /usr/share/mime || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
fi

exit 0
