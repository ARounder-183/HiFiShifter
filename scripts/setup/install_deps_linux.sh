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
sudo apt-get install -y pkg-config cmake libglib2.0-dev libgirepository1.0-dev gobject-introspection libssl-dev build-essential libgtk-3-dev libgdk-pixbuf2.0-dev libasound2-dev libfuse2 fuse || true

# Additional dev packages commonly required by linuxdeploy and the GTK/GStreamer plugins
# (helps Tauri's bundler find webkit/gstreamer and related girs at build time)
sudo apt-get install -y \
  libwebkit2gtk-4.1-dev libjavascriptcoregtk-4.1-dev gir1.2-webkit2-4.1 gir1.2-javascriptcoregtk-4.1 \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libx11-dev libx11-xcb-dev libxkbcommon-dev libpango1.0-dev libcairo2-dev libatk1.0-dev \
  libdbus-1-dev desktop-file-utils librsvg2-dev || true

# Try to install explicit webkit/javascriptcore -dev packages (common names)
echo "[install_deps_linux] Installing webkit2gtk and javascriptcore dev packages"
sudo apt-get install -y libwebkit2gtk-4.1-dev libjavascriptcoregtk-4.1-dev gir1.2-webkit2-4.1 gir1.2-javascriptcoregtk-4.1 || true

# libsoup package name varies across distributions; try common variants
sudo apt-get install -y libsoup3.0-dev || sudo apt-get install -y libsoup-3.0-dev || sudo apt-get install -y libsoup2.4-dev || echo '[install_deps_linux] libsoup dev package not found; continuing'

# webkit2gtk / GIR packages fallback (additional variants)
sudo apt-get install -y libwebkit2gtk-4.0-dev libwebkit2gtk-dev gir1.2-webkit2-4.0 || true

# Ensure PKG_CONFIG_PATH available to subsequent steps (in CI append to GITHUB_ENV)
PKG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/pkgconfig:/usr/share/pkgconfig"
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

echo "[install_deps_linux] Ensure Node is available and new enough for the frontend"
# `frontend/package.json` pins vite 8 / vitest 5. Their engines are
# `^20.19.0 || >=22.12.0` and `^22.12.0 || ^24.0.0 || >=26.0.0`. Ubuntu noble
# ships Node 18, which dies inside rolldown with
# "The requested module 'node:util' does not provide an export named 'styleText'".
# A bare `command -v node` therefore is not a sufficient check.
NODE_MIN_MAJOR=22
NODE_MIN_MINOR=12
NODE_INSTALL_VERSION="${NODE_INSTALL_VERSION:-v24.22.1}"
node_is_new_enough() {
  command -v node >/dev/null 2>&1 || return 1
  local major minor
  major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null)" || return 1
  minor="$(node -p 'process.versions.node.split(".")[1]' 2>/dev/null)" || return 1
  if [ "${major:-0}" -gt "$NODE_MIN_MAJOR" ]; then return 0; fi
  if [ "${major:-0}" -eq "$NODE_MIN_MAJOR" ] && [ "${minor:-0}" -ge "$NODE_MIN_MINOR" ]; then return 0; fi
  return 1
}

if ! node_is_new_enough; then
  echo "[install_deps_linux] Node is missing or older than ${NODE_MIN_MAJOR}.${NODE_MIN_MINOR} (found: $(node --version 2>/dev/null || echo none))"
  echo "[install_deps_linux] Installing Node ${NODE_INSTALL_VERSION} into \$HOME/.local/opt (no sudo required)"
  # 【为什么按架构分发】Node 官方 tarball 的目录名带架构后缀；在 ARM64 Linux
  # 上硬编码 linux-x64 会装错二进制（执行时 "cannot execute binary file"）。
  case "$(uname -m)" in
    aarch64|arm64) NODE_ARCH="arm64" ;;
    *)             NODE_ARCH="x64" ;;
  esac
  NODE_TARBALL_DIR="node-${NODE_INSTALL_VERSION}-linux-${NODE_ARCH}"
  mkdir -p "$HOME/.local/opt"
  curl -fSL --retry 3 -o "$HOME/.local/opt/node.tar.xz" \
    "https://nodejs.org/dist/${NODE_INSTALL_VERSION}/${NODE_TARBALL_DIR}.tar.xz"
  rm -rf "$HOME/.local/opt/${NODE_TARBALL_DIR}"
  mkdir -p "$HOME/.local/opt/${NODE_TARBALL_DIR}"
  tar -xJf "$HOME/.local/opt/node.tar.xz" -C "$HOME/.local/opt/${NODE_TARBALL_DIR}" --strip-components=1
  rm -f "$HOME/.local/opt/node.tar.xz"
  ln -sfn "$NODE_TARBALL_DIR" "$HOME/.local/opt/node"
  export PATH="$HOME/.local/opt/node/bin:$PATH"
  echo "[install_deps_linux] Add this to your shell rc so later shells see it:"
  echo '  export PATH="$HOME/.local/opt/node/bin:$PATH"'
fi
node --version

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
    URL="https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-aarch64.AppImage"
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
