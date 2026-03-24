#!/usr/bin/env bash
set -euo pipefail

echo "[install_deps_linux] Updating apt and installing development packages"
sudo apt-get update -y || true

# Ensure universe repository (some webkit/dev packages live there)
echo "[install_deps_linux] Ensuring 'universe' apt repository is enabled"
sudo apt-get install -y software-properties-common || true
sudo add-apt-repository -y universe || true
sudo apt-get update -y || true

# Core packages used by native crates
# Also install libfuse2 and fuse so AppImages can run in CI (provides libfuse.so.2)
sudo apt-get install -y pkg-config libglib2.0-dev libgirepository1.0-dev gobject-introspection libssl-dev build-essential libgtk-3-dev libgdk-pixbuf2.0-dev libasound2-dev libfuse2 fuse || true

# Additional dev packages commonly required by linuxdeploy and the GTK/GStreamer plugins
# (helps Tauri's bundler find webkit/gstreamer and related girs at build time)
sudo apt-get install -y \
  libwebkit2gtk-4.1-dev libjavascriptcoregtk-4.1-dev gir1.2-webkit2-4.1 gir1.2-javascriptcoregtk-4.1 \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libx11-dev libx11-xcb-dev libxkbcommon-dev libpango1.0-dev libcairo2-dev libatk1.0-dev \
  libdbus-1-dev desktop-file-utils || true

# Try to install explicit webkit/javascriptcore -dev packages (common names)
echo "[install_deps_linux] Installing webkit2gtk and javascriptcore dev packages"
sudo apt-get install -y libwebkit2gtk-4.1-dev libjavascriptcoregtk-4.1-dev gir1.2-webkit2-4.1 gir1.2-javascriptcoregtk-4.1 || true

# libsoup package name varies across distributions; try common variants
sudo apt-get install -y libsoup3.0-dev || sudo apt-get install -y libsoup-3.0-dev || sudo apt-get install -y libsoup2.4-dev || echo '[install_deps_linux] libsoup dev package not found; continuing'

# webkit2gtk / GIR packages fallback (additional variants)
sudo apt-get install -y libwebkit2gtk-4.0-dev libwebkit2gtk-dev gir1.2-webkit2-4.0 || true

# Ensure PKG_CONFIG_PATH available to subsequent steps (in CI append to GITHUB_ENV)
PKG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/pkgconfig:/usr/share/pkgconfig:/usr/lib/pkgconfig"
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "PKG_CONFIG_PATH=$PKG_PATH" >> "$GITHUB_ENV"
else
  export PKG_CONFIG_PATH="$PKG_PATH"
fi

echo "[install_deps_linux] Checking pkg-config availability"
PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --modversion gobject-2.0 || true
PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --modversion gio-2.0 || true
PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --modversion glib-2.0 || true
PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --modversion gdk-3.0 || true
PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --modversion libsoup-3.0 || PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --modversion libsoup-2.4 || true
PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --modversion javascriptcoregtk-4.1 || true
PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --modversion webkit2gtk-4.1 || true
PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --modversion alsa || true

if ! PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --modversion webkit2gtk-4.1 >/dev/null 2>&1; then
  echo "[install_deps_linux] webkit2gtk-4.1 not found by pkg-config after attempted installs"
  echo "[install_deps_linux] Listing dpkg info for candidate packages (if installed)"
  dpkg -l | grep webkit || true
  dpkg -l | grep javascriptcore || true
  echo "[install_deps_linux] Listing /usr/lib*/pkgconfig for webkit2gtk pc files"
  ls -la /usr/lib/*/pkgconfig | grep webkit || true
  ls -la /usr/lib*/pkgconfig | grep webkit || true
  echo "[install_deps_linux] Showing pkg-config search path: $PKG_CONFIG_PATH"
fi

echo "[install_deps_linux] Ensure Node is available (install if missing)"
if ! command -v node >/dev/null 2>&1; then
  echo "[install_deps_linux] Installing nodejs/npm from apt repositories"
  sudo apt-get install -y nodejs npm
fi

if [ "${SKIP_FRONTEND:-0}" != "1" ]; then
  echo "[install_deps_linux] Installing frontend npm deps"
  npm --prefix frontend ci
else
  echo "[install_deps_linux] SKIP_FRONTEND set; skipping frontend npm install"
fi

echo "[install_deps_linux] Done"

echo "[install_deps_linux] Installing appimagetool for bundling (if missing)"
APPIMAGETOOL_DIR=/usr/local/bin
if ! command -v appimagetool >/dev/null 2>&1; then
  ARCH=$(uname -m || echo x86_64)
  echo "[install_deps_linux] detected arch: $ARCH"
  if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then
    URL="https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
  elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    URL="https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-aarch64.AppImage" || true
  else
    URL="https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
  fi
  echo "[install_deps_linux] Downloading appimagetool from $URL"
  sudo curl -fsSL -o $APPIMAGETOOL_DIR/appimagetool "$URL" || true
  sudo chmod +x $APPIMAGETOOL_DIR/appimagetool || true
fi

if command -v appimagetool >/dev/null 2>&1; then
  echo "[install_deps_linux] appimagetool installed at $(which appimagetool)"
else
  echo "[install_deps_linux] WARNING: appimagetool not available; Linux bundling may fail"
fi

echo "[install_deps_linux] Installing additional runtime packages needed by linuxdeploy/appimage bundler"
# Install runtime libs commonly required by linuxdeploy and plugins
sudo apt-get install -y patchelf xz-utils wget libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 libgtk-3-0 libdbus-1-3 libcanberra-gtk3-0 || true

echo "[install_deps_linux] Ensuring required tools for running AppImages without FUSE"
sudo apt-get install -y squashfs-tools || true

echo "[install_deps_linux] Ensure xdg-utils (provides xdg-open) is available for bundlers"
sudo apt-get install -y xdg-utils || true

echo "[install_deps_linux] Done installing extra runtime deps"

# Ensure modern FUSE implementation available (install fuse3 and libfuse2)
echo "[install_deps_linux] Ensuring fuse3 and libfuse2 are installed"
sudo apt-get install -y fuse3 libfuse2 || true
