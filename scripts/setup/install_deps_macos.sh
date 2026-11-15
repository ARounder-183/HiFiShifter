#!/usr/bin/env bash
set -euo pipefail

echo "[install_deps_macos] Ensure Homebrew is installed"
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Installing..."
  # 【为什么不用命令替换】`bash -c "$(curl ...)"` 在 curl 失败时替换成空串，
  # `bash -c ""` 退出 0 —— set -e 抓不到，Homebrew 静默没装，后面每个 brew
  # 命令才逐个报 "command not found"。先落盘、显式查退出码、再执行。
  BREW_INSTALLER="$(mktemp -t hfs-brew-install.XXXXXX.sh)"
  trap 'rm -f "$BREW_INSTALLER"' EXIT
  curl -fsSL -o "$BREW_INSTALLER" https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh
  if [ ! -s "$BREW_INSTALLER" ]; then
    echo "[install_deps_macos] FATAL: Homebrew installer download is empty" >&2
    exit 1
  fi
  /bin/bash "$BREW_INSTALLER"
fi

echo "[install_deps_macos] Installing Homebrew packages"

# GitHub macOS runner images preinstall stale third-party taps (e.g.
# hashicorp/tap, aws/tap) whose formulae fail to import during `brew update`
# on recent Homebrew ("hashicorp/tap/vagrant: formula requires at least a
# URL"). Untap every non-official tap first; nothing this script installs
# comes from them.
while IFS= read -r tap; do
  [ -z "$tap" ] && continue
  case "$tap" in
    homebrew/*) continue ;;
  esac
  echo "[install_deps_macos] Untapping third-party tap: $tap"
  brew untap --force "$tap" || true
done < <(brew tap 2>/dev/null || true)

brew update || true
brew install cmake || true

# Install rustup if the user does not already have a Rust toolchain.
if ! command -v cargo >/dev/null 2>&1 && ! command -v rustup >/dev/null 2>&1; then
  brew install rustup || true
fi

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
