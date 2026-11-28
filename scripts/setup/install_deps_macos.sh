#!/usr/bin/env bash
set -euo pipefail

echo "[install_deps_macos] Ensure Homebrew is installed"
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Installing..."
  # [Why not command substitution] If curl fails, `bash -c "$(curl ...)"` becomes
  # `bash -c ""`, which exits 0 -- set -e never catches it, Homebrew silently fails
  # to install, and only later does every brew command report "command not found"
  # one by one. Write to disk first, check the exit code explicitly, then execute.
  BREW_INSTALLER="$(mktemp -t hfs-brew-install.XXXXXX.sh)"
  trap 'rm -f "$BREW_INSTALLER"' EXIT
  curl -fsSL -o "$BREW_INSTALLER" https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh
  if [ ! -s "$BREW_INSTALLER" ]; then
    echo "[install_deps_macos] FATAL: Homebrew installer download is empty" >&2
    exit 1
  fi
  /bin/bash "$BREW_INSTALLER"
fi

echo "[install_deps_macos] Checking for the tools this script would install"

# [Why these two variables] The runner images already ship cmake, node and a Rust toolchain.
# Homebrew's default behaviour is to update itself before `brew install`, and on Intel macOS
# (Tier 3: no bottles) "upgrading" an already-installed formula is a **source build**: run
# 46032661579 spent 34.6 minutes compiling cmake 4.5.1 over a perfectly usable 4.4.2. With
# auto-update off, an install only happens when something is genuinely missing.
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1

need_install=0
command -v cmake >/dev/null 2>&1 || { echo "[install_deps_macos] cmake missing"; need_install=1; }
command -v node >/dev/null 2>&1 || { echo "[install_deps_macos] node missing"; need_install=1; }
if ! command -v cargo >/dev/null 2>&1 && ! command -v rustup >/dev/null 2>&1; then
  echo "[install_deps_macos] no Rust toolchain"
  need_install=1
fi

if [ "$need_install" = 1 ]; then
  echo "[install_deps_macos] Installing Homebrew packages"

  # GitHub macOS runner images preinstall stale third-party taps (e.g.
  # hashicorp/tap, aws/tap) whose formulae fail to import during `brew update`
  # on recent Homebrew ("hashicorp/tap/vagrant: formula requires at least a
  # URL"). Untap every non-official tap first; nothing this script installs
  # comes from them. [Why only inside this branch] The untap loop exists to make `brew update`
  # work; when nothing is installed, no Homebrew command runs at all.
  while IFS= read -r tap; do
    [ -z "$tap" ] && continue
    case "$tap" in
      homebrew/*) continue ;;
    esac
    echo "[install_deps_macos] Untapping third-party tap: $tap"
    brew untap --force "$tap" || true
  done < <(brew tap 2>/dev/null || true)

  brew update || true
  command -v cmake >/dev/null 2>&1 || brew install cmake || true
  # Install rustup if the user does not already have a Rust toolchain.
  if ! command -v cargo >/dev/null 2>&1 && ! command -v rustup >/dev/null 2>&1; then
    brew install rustup || true
  fi
  command -v node >/dev/null 2>&1 || brew install node || true
else
  echo "[install_deps_macos] cmake, node and a Rust toolchain are already present; nothing to install"
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

if [ "${SKIP_FRONTEND:-0}" != "1" ]; then
  echo "[install_deps_macos] Installing frontend npm deps"
  npm --prefix frontend ci
else
  echo "[install_deps_macos] SKIP_FRONTEND set; skipping frontend npm install"
fi

echo "[install_deps_macos] Done"
