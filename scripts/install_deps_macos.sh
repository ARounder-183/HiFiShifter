#!/usr/bin/env bash
set -euo pipefail

echo "[install_deps_macos] Ensure Homebrew is installed"
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Installing..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "[install_deps_macos] Installing Homebrew packages"
brew update || true

# Ensure PKG_CONFIG_PATH available to subsequent steps (in CI append to GITHUB_ENV)
PKG_PATH="/opt/homebrew/lib/pkgconfig:/opt/homebrew/share/pkgconfig:/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig"
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "PKG_CONFIG_PATH=$PKG_PATH" >> "$GITHUB_ENV"
else
  export PKG_CONFIG_PATH="$PKG_PATH"
fi

echo "[install_deps_macos] Checking pkg-config availability"
PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --modversion gobject-2.0 || true
PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --modversion gio-2.0 || true
PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --modversion glib-2.0 || true

echo "[install_deps_macos] Ensure Node is available (install if missing)"
if ! command -v node >/dev/null 2>&1; then
  brew install node || true
fi

if [ "${SKIP_FRONTEND:-0}" != "1" ]; then
  echo "[install_deps_macos] Installing frontend npm deps"
  npm --prefix frontend ci
else
  echo "[install_deps_macos] SKIP_FRONTEND set; skipping frontend npm install"
fi

echo "[install_deps_macos] Done"

echo "[install_deps_macos] Note: create-dmg global install is handled by CI after Node setup"
