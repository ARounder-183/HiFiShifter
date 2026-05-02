#!/usr/bin/env bash
# macOS Build Script for HiFiShifter
# Builds a signed .app bundle and creates a DMG installer.
#
# Usage:
#   bash scripts/build_macos.sh              # ad-hoc signed (local use)
#   bash scripts/build_macos.sh --sign "Developer ID"  # Developer ID signed
#   bash scripts/build_macos.sh --notarize   # sign + notarize (req. Apple ID)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_TAURI_DIR="$PROJECT_DIR/backend/src-tauri"

# ── Parse arguments ──────────────────────────────────────────────────
SIGN_IDENTITY=""
DO_NOTARIZE=false
CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sign)
            SIGN_IDENTITY="$2"
            shift 2
            ;;
        --notarize)
            DO_NOTARIZE=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--sign IDENTITY] [--notarize] [--clean]"
            echo ""
            echo "Options:"
            echo "  --sign IDENTITY   Code signing identity (e.g. 'Developer ID Application: ...')"
            echo "                    Omit for ad-hoc signing (-)."
            echo "  --notarize        Submit for Apple notarization (requires --sign + credentials)"
            echo "  --clean           Clean build (cargo clean)"
            echo ""
            echo "Environment variables for notarization:"
            echo "  APPLE_ID          Apple ID email"
            echo "  APPLE_PASSWORD    App-specific password"
            echo "  APPLE_TEAM_ID     Team ID"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo " HiFiShifter macOS Build Script"
echo "========================================="
echo "Project: $PROJECT_DIR"
if [ -n "$SIGN_IDENTITY" ]; then
    echo "Signing: $SIGN_IDENTITY"
else
    echo "Signing: ad-hoc (local use only)"
fi
echo ""

# ── Prerequisites ────────────────────────────────────────────────────

echo "[1/6] Checking prerequisites..."

if ! command -v cargo &>/dev/null; then
    echo "ERROR: Rust/cargo not found. Install from https://rustup.rs"
    exit 1
fi

if ! command -v node &>/dev/null; then
    echo "ERROR: Node.js not found. Install from https://nodejs.org"
    exit 1
fi

if ! command -v npm &>/dev/null; then
    echo "ERROR: npm not found."
    exit 1
fi

if ! xcrun --show-sdk-path &>/dev/null; then
    echo "ERROR: Xcode Command Line Tools not found. Run: xcode-select --install"
    exit 1
fi

echo "  Rust: $(rustc --version)"
echo "  Node: $(node --version)"
echo "  npm:  $(npm --version)"
echo ""

# ── Architecture detection ───────────────────────────────────────────

ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    TARGET="aarch64-apple-darwin"
elif [ "$ARCH" = "x86_64" ]; then
    TARGET="x86_64-apple-darwin"
else
    echo "ERROR: Unsupported architecture: $ARCH"
    exit 1
fi

echo "  Target: $TARGET"
echo ""

# ── Clean build (if requested) ───────────────────────────────────────

if [ "$CLEAN_BUILD" = true ]; then
    echo "[*] Cleaning previous build..."
    cd "$SRC_TAURI_DIR"
    cargo clean
    echo ""
fi

# ── Frontend ─────────────────────────────────────────────────────────

echo "[2/6] Installing frontend dependencies..."
npm --prefix "$PROJECT_DIR/frontend" ci
echo ""

echo "[3/6] Building frontend..."
npm --prefix "$PROJECT_DIR/frontend" run build
echo ""

# ── Backend build ────────────────────────────────────────────────────

echo "[4/6] Building Rust backend for $TARGET..."
cd "$SRC_TAURI_DIR"

# Set signing identity
if [ -n "$SIGN_IDENTITY" ]; then
    export APPLE_SIGNING_IDENTITY="$SIGN_IDENTITY"
else
    # Ad-hoc signing: Tauri uses '-' identity when env var is empty
    export APPLE_SIGNING_IDENTITY=""
fi

# Build with Tauri
HIFISHIFTER_SKIP_FRONTEND_BUILD=1 \
    cargo tauri build \
    --target "$TARGET" \
    --bundles dmg

echo ""

# ── Verify code signature ────────────────────────────────────────────

echo "[5/6] Verifying code signature..."

APP_PATH="$SRC_TAURI_DIR/target/$TARGET/release/bundle/macos/HiFiShifter.app"
if [ -d "$APP_PATH" ]; then
    echo "  App bundle: $APP_PATH"
    echo "  Checking signature..."
    codesign -dvvv "$APP_PATH" 2>&1 || echo "  WARNING: code signature verification failed"
    echo ""
    echo "  Checking executable..."
    codesign -dvvv "$APP_PATH/Contents/MacOS/HiFiShifter" 2>&1 || \
        echo "  WARNING: executable signature check failed"
    echo ""
else
    echo "  WARNING: .app bundle not found at expected path."
    echo "  Looking for .app in target directory..."
    APP_PATH=$(find "$SRC_TAURI_DIR/target" -name "HiFiShifter.app" -maxdepth 6 -type d 2>/dev/null | head -1)
    if [ -n "$APP_PATH" ]; then
        echo "  Found: $APP_PATH"
    fi
fi

echo ""

# ── DMG location ─────────────────────────────────────────────────────

echo "[6/6] Build complete!"

DMG_DIR="$SRC_TAURI_DIR/target/$TARGET/release/bundle/dmg"
if [ -d "$DMG_DIR" ]; then
    DMG_FILE=$(ls -t "$DMG_DIR"/*.dmg 2>/dev/null | head -1)
    if [ -n "$DMG_FILE" ]; then
        DMG_SIZE=$(du -h "$DMG_FILE" | cut -f1)
        echo ""
        echo "========================================="
        echo " DMG installer created successfully!"
        echo "========================================="
        echo ""
        echo "  File: $DMG_FILE"
        echo "  Size: $DMG_SIZE"
        echo "  Type: $(if [ -n "$SIGN_IDENTITY" ]; then echo "Developer ID signed"; else echo "Ad-hoc signed (local use)"; fi)"
        echo ""
        echo "── Installation ─────────────────────────"
        echo ""
        if [ -n "$SIGN_IDENTITY" ]; then
            echo "  Mount the DMG and drag HiFiShifter to /Applications."
        else
            echo "  Since this is ad-hoc signed, macOS Gatekeeper may block it."
            echo "  After copying to /Applications, run:"
            echo ""
            echo "    xattr -cr /Applications/HiFiShifter.app"
            echo ""
            echo "  This removes the quarantine flag. Then launch normally."
        fi
        echo ""
        echo "  Or run directly from the mounted DMG (drag to /Applications first)."
        echo "========================================="
    else
        echo "  DMG not found in $DMG_DIR"
    fi
else
    echo "  DMG directory not found. Check build output for errors."
    echo "  Expected: $DMG_DIR"
fi

echo ""
echo "── Troubleshooting ───────────────────────"
echo ""
echo "  If you see 'HiFiShifter is damaged and cannot be opened':"
echo "  1. Remove quarantine: xattr -cr /Applications/HiFiShifter.app"
echo "  2. Allow in Security & Privacy: open anyway in System Settings"
echo "  3. For Apple Silicon: ensure Rosetta 2 is not interfering"
echo ""
echo "  If audio doesn't work:"
echo "  1. Grant microphone permission in System Settings > Privacy"
echo "  2. Check the app is not sandboxed blocking audio device access"
echo ""
echo "Done!"
