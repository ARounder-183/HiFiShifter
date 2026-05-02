#!/usr/bin/env bash
# Fix "HiFiShifter is damaged and cannot be opened" error on macOS.
#
# This happens because macOS Gatekeeper quarantines apps downloaded
# from the internet that aren't notarized by Apple.
#
# This script:
#   1. Removes the quarantine extended attribute
#   2. Re-checks the code signature
#   3. Provides next steps
set -euo pipefail

APP_PATH="${1:-/Applications/HiFiShifter.app}"

echo "========================================="
echo " HiFiShifter macOS Fix Tool"
echo "========================================="
echo ""

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: HiFiShifter.app not found at: $APP_PATH"
    echo ""
    echo "Usage: $0 [path/to/HiFiShifter.app]"
    echo ""
    echo "Make sure you've copied HiFiShifter to /Applications first."
    exit 1
fi

echo "Target: $APP_PATH"
echo ""

# ── Check current quarantine status ───────────────────────────────────

echo "[1/4] Checking quarantine status..."
QUARANTINE=$(xattr -l "$APP_PATH" 2>/dev/null | grep "com.apple.quarantine" || true)
if [ -n "$QUARANTINE" ]; then
    echo "  Quarantine flag found:"
    echo "  $QUARANTINE"
else
    echo "  No quarantine flag found."
fi
echo ""

# ── Check code signature ─────────────────────────────────────────────

echo "[2/4] Checking code signature..."
SIG_INFO=$(codesign -dvvv "$APP_PATH" 2>&1) || true
if echo "$SIG_INFO" | grep -q "Signature="; then
    echo "  App is signed."
    echo "  $(echo "$SIG_INFO" | grep "Authority\|Signature\|TeamIdentifier" | head -5)"
else
    echo "  WARNING: App does not appear to be signed."
    echo "  This is unusual - the build process should ad-hoc sign it."
fi
echo ""

# ── Remove quarantine ────────────────────────────────────────────────

echo "[3/4] Removing quarantine flag..."
xattr -cr "$APP_PATH"
echo "  Quarantine removed from $APP_PATH"

# Also fix embedded frameworks/dylibs
find "$APP_PATH/Contents" -type f -name "*.dylib" 2>/dev/null | while read -r dylib; do
    xattr -cr "$dylib" 2>/dev/null || true
done

# Also fix bundled resources (ONNX models)
find "$APP_PATH/Contents/Resources" -type f 2>/dev/null | while read -r res; do
    xattr -cr "$res" 2>/dev/null || true
done

echo "  All nested files processed."
echo ""

# ── Re-verify ────────────────────────────────────────────────────────

echo "[4/4] Re-verifying..."
QUARANTINE_AFTER=$(xattr -l "$APP_PATH" 2>/dev/null | grep "com.apple.quarantine" || true)
if [ -z "$QUARANTINE_AFTER" ]; then
    echo "  Quarantine successfully removed."
else
    echo "  WARNING: Quarantine may still be present."
fi
echo ""

echo "========================================="
echo " Fix applied!"
echo "========================================="
echo ""
echo "Now try opening HiFiShifter. If it still fails:"
echo ""
echo "  1. Open System Settings > Privacy & Security"
echo "  2. Scroll down to the Security section"
echo "  3. You should see a message about HiFiShifter being blocked"
echo "  4. Click 'Open Anyway'"
echo ""
echo "  Or run:"
echo "    sudo spctl --master-disable   (NOT recommended)"
echo ""
echo "  To verify the complete fix:"
echo "    spctl --assess --verbose /Applications/HiFiShifter.app"
echo ""
