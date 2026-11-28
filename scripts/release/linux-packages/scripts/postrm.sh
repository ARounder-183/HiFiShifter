#!/bin/sh
# Post-remove hook shared by the deb and the rpm (nfpm maps it to %postun and to postrm).
#
# [Why it only refreshes caches] The files a package owns are removed by the package manager
# itself, from the file list it recorded at install time; a hand-written `rm -rf` here could only
# delete something that is not ours. In particular $HOME is never touched: settings, projects,
# audio files and models-user survive both `apt remove` and `apt purge`, which is the one promise
# an uninstall must keep.
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
