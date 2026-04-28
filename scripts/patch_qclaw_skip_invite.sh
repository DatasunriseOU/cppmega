#!/usr/bin/env bash
# Patch QClaw to skip invite code verification.
# Run on macOS. Requires Node.js >= 22.
set -euo pipefail

APP="${QCLAW_APP:-/Applications/QClaw.app}"
ASAR="$APP/Contents/Resources/app.asar"
ELECTRON_FW="$APP/Contents/Frameworks/Electron Framework.framework/Versions/A/Electron Framework"

echo "==> Checking prerequisites..."
command -v node >/dev/null 2>&1 || { echo "FATAL: node not found"; exit 1; }
NODE_MAJOR=$(node -e 'console.log(process.versions.node.split(".")[0])')
if [ "$NODE_MAJOR" -lt 22 ]; then
  echo "FATAL: Node.js >= 22 required, got $NODE_MAJOR"
  exit 1
fi

if [ ! -d "$APP" ]; then
  echo "FATAL: $APP not found. Set QCLAW_APP=/path/to/QClaw.app"
  exit 1
fi
if [ ! -f "$ASAR" ]; then
  echo "FATAL: $ASAR not found"
  exit 1
fi

# Kill QClaw if running
if pgrep -f QClaw >/dev/null 2>&1; then
  echo "==> Stopping QClaw..."
  pkill -f QClaw || true
  sleep 2
fi

WORKDIR=$(mktemp -d /tmp/qclaw_patch.XXXXXX)
EXTRACT="$WORKDIR/app_extracted"
trap 'rm -rf "$WORKDIR"' EXIT

echo "==> Extracting app.asar..."
npx --yes @electron/asar extract "$ASAR" "$EXTRACT"

echo "==> Finding invite check..."
TARGET_FILE=""
FOUND_LINE=""
while IFS= read -r -d '' f; do
  if grep -q "inviteCodeVerified" "$f" 2>/dev/null; then
    FOUND_LINE=$(grep "inviteCodeVerified" "$f" | head -1)
    TARGET_FILE="$f"
    break
  fi
done < <(find "$EXTRACT" -name '*.js' -print0 2>/dev/null || true)

if [ -z "$TARGET_FILE" ]; then
  echo "FATAL: could not find inviteCodeVerified in extracted JS"
  find "$EXTRACT" -name '*.js' -exec grep -l "nvi" {} \; 2>/dev/null | head -5 || true
  echo "Try searching for the pattern manually:"
  echo "  grep -r 'invite' $EXTRACT"
  exit 1
fi

echo "    file: $TARGET_FILE"
echo "    match: $FOUND_LINE"

# Extract the variable name used for inviteCodeVerified.
# Minified JS pattern: {...,inviteCodeVerified:ABC,...}  and later: ABC=!1
VAR_NAME=$(echo "$FOUND_LINE" | perl -ne 'print $1 if /inviteCodeVerified\s*:\s*(\w+)/')
if [ -z "$VAR_NAME" ]; then
  # Fallback: try looking for the key-value pair differently (maybe it's a quoted key)
  VAR_NAME=$(echo "$FOUND_LINE" | perl -ne 'print $1 if /"inviteCodeVerified"\s*:\s*(\w+)/')
fi
if [ -z "$VAR_NAME" ]; then
  echo "FATAL: could not extract variable name from: $FOUND_LINE"
  exit 1
fi
echo "    variable: $VAR_NAME"

# Patch from !1 (false) to !0 (true)
if grep -q "${VAR_NAME}=!1" "$TARGET_FILE"; then
  perl -pi -e "s/\b${VAR_NAME}=!1\b/${VAR_NAME}=!0/g" "$TARGET_FILE"
  echo "==> Patched $VAR_NAME from !1 to !0"
elif grep -q "${VAR_NAME}=!0" "$TARGET_FILE"; then
  echo "==> Already patched ($VAR_NAME=!0), skipping"
else
  echo "FATAL: could not find ${VAR_NAME}=!1 or ${VAR_NAME}=!0"
  # Show context around the variable to help debug
  grep -o ".{0,40}${VAR_NAME}.{0,40}" "$TARGET_FILE" | head -5 || true
  exit 1
fi

echo "==> Repacking app.asar..."
cp "$ASAR" "$ASAR.bak"
npx --yes @electron/asar pack "$EXTRACT" "$ASAR"

echo "==> Patching Electron ASAR integrity fuse..."
# The Electron Framework binary contains a fuse wire at a known offset or marked
# by a magic string.  Recent Electron uses "dL-L" as the marker; older versions
# use a uint32 magic.  We disable the ASAR integrity fuse so the repacked archive
# loads without a matching hash.
python3 - "$ELECTRON_FW" <<'PYEOF'
import sys, struct

binary = sys.argv[1]
data = bytearray(open(binary, 'rb').read())

# Search for known fuse magic markers.  Recent Electron uses "dL-L";
# older versions use a uint32 LE magic like 0xFEEDC0DE.
magic_bytes = None
for candidate in (b"dL-L", struct.pack('<I', 0xFEEDC0DE), struct.pack('<I', 0xFEEDC0DF)):
    idx = data.find(candidate)
    if idx != -1:
        magic_bytes = candidate
        print(f"fuse magic {candidate!r} at offset {idx:#x}")
        break

if magic_bytes is None:
    print("WARNING: could not find Electron fuse magic marker")
    print("The ASAR integrity check may reject the patched archive.")
    print("If QClaw crashes on launch, try:")
    print("  https://github.com/warm-mannalichen723/qclaw-skip-invite")
    sys.exit(0)

fuse_start = idx + len(magic_bytes)
print(f"fuse region at offset {fuse_start:#x}")

# Fuse wire: magic (4B) + version (2B) + padding (2B) + fuses[]
# After magic bytes: skip version+padding (4 bytes total), then fuses start.
# ASAR integrity is typically fuse index 3 (0-indexed).
# Try offsets 3, 4, and 5 to be safe.
patched = False
for offset in (3, 4, 5):
    pos = fuse_start + offset
    if pos >= len(data):
        break
    if data[pos] == 0x31:  # '1' = enabled
        data[pos] = 0x30    # '0' = disabled
        print(f"asar fuse at offset {pos:#x}: 0x31 -> 0x30 (disabled)")
        patched = True
        break

if not patched:
    # Show what's there
    sample = ' '.join(f'0x{b:02x}' for b in data[fuse_start:fuse_start+12])
    print(f"fuse bytes: {sample}")
    print("WARNING: could not find ASAR integrity fuse byte (0x31)")
    print("The fuse layout may differ in this Electron version.")

open(binary, 'wb').write(data)
PYEOF

echo "==> Re-signing..."
codesign --force --deep --sign - "$APP"

echo ""
echo "==> Done. QClaw invite check patched."
echo "    Backup saved at $ASAR.bak"
echo "    Launch QClaw — invite screen should be skipped."
