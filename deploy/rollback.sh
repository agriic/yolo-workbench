#!/usr/bin/env bash
# Switch the YOLO Workbench server to another installed release: back to the one before after an upgrade,
# or to any release in /opt/yolo-workbench/releases/. Run as root on the server:
#
#   sudo bash deploy/rollback.sh               # the previous release (/opt/yolo-workbench/previous)
#   sudo bash deploy/rollback.sh 2026.10.21    # or name one
#   bash deploy/rollback.sh --list             # the installed releases
#
# It runs `check` with that release's code first (nothing changes when it fails, unless --force), points
# /opt/yolo-workbench/current at it and `previous` at the release it leaves (so running it again returns),
# puts back that release's systemd unit, and restarts the service: the active training run pauses at its
# next checkpoint first, which can take up to [training] pause_deadline_s. Then it waits for /readyz.
# The projects folder and /etc/yolo-workbench/ are not touched.
set -euo pipefail

PREFIX=/opt/yolo-workbench
ACCOUNT=yolo-workbench
CONFIG=/etc/yolo-workbench/config.toml
SERVICE=yolo-workbench
NAME_PATTERN='^[A-Za-z0-9][A-Za-z0-9._+-]*$'  # install.sh's version names

usage() { echo "Usage: sudo bash $0 [<release>] [--force]   |   $0 --list"; }

list() {
  local current previous dir marks installed
  current="$(readlink -f "$PREFIX/current" 2>/dev/null || true)"
  previous="$(readlink -f "$PREFIX/previous" 2>/dev/null || true)"
  echo "Installed releases in $PREFIX/releases (newest first):"
  while IFS= read -r dir; do
    dir="$(readlink -f "$dir")"
    installed="$(sed -n 's/.*"installed": *"\([^"]*\)".*/\1/p' "$dir/release.json" 2>/dev/null || true)"
    marks=""
    if [ "$dir" = "$current" ]; then marks="$marks (current)"; fi
    if [ "$dir" = "$previous" ]; then marks="$marks (previous)"; fi
    printf '  %-40s %-20s%s\n' "$(basename "$dir")" "${installed:-}" "$marks"
  done < <(ls -1td "$PREFIX"/releases/*/ 2>/dev/null)
}

FORCE=0
WANTED=""
for arg in "$@"; do
  case "$arg" in
    --list) list; exit 0 ;;
    --force) FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) usage >&2; exit 2 ;;
    *) if [ -n "$WANTED" ]; then usage >&2; exit 2; fi; WANTED="$arg" ;;
  esac
done
if [ "$(id -u)" -ne 0 ]; then echo "Run as root (sudo bash $0)" >&2; exit 1; fi
cd /  # `check` runs as the service account, which may not read the caller's folder

CURRENT="$(readlink -f "$PREFIX/current" 2>/dev/null || true)"
if [ -n "$WANTED" ]; then
  if [[ ! "$WANTED" =~ $NAME_PATTERN ]]; then echo "Invalid release name '$WANTED'" >&2; exit 2; fi
  TARGET="$(readlink -f "$PREFIX/releases/$WANTED" 2>/dev/null || true)"
elif [ -L "$PREFIX/previous" ]; then
  WANTED=previous
  TARGET="$(readlink -f "$PREFIX/previous" 2>/dev/null || true)"
else
  echo "No previous release is recorded; name one:" >&2
  list >&2
  exit 1
fi
SERVER="$TARGET/.venv/bin/yolo-workbench-server"
if [ -z "$TARGET" ] || [ ! -x "$SERVER" ] || [ ! -f "$TARGET/deploy/$SERVICE.service" ]; then
  echo "$WANTED is not an installed release" >&2
  list >&2
  exit 1
fi
NAME="$(basename "$TARGET")"
if [ "$TARGET" = "$CURRENT" ]; then echo "$NAME is already the active release" >&2; exit 1; fi

echo "Checking the configuration with $NAME:"
if CHECKED="$(runuser -u "$ACCOUNT" -- "$SERVER" check --config "$CONFIG" 2>&1)"; then STATUS=0; else STATUS=$?; fi
echo "$CHECKED"
if [ "$STATUS" -ne 0 ]; then
  if [ "$FORCE" -eq 0 ]; then
    echo "The check failed with $NAME (above); nothing was changed. Fix what it reports, or pass --force to switch anyway." >&2
    exit 1
  fi
  echo "The check failed; switching anyway (--force)." >&2
fi

if [ -n "$CURRENT" ]; then
  ln -sfn "$CURRENT" "$PREFIX/previous.new"
  mv -T "$PREFIX/previous.new" "$PREFIX/previous"
fi
ln -sfn "$TARGET" "$PREFIX/current.new"
mv -T "$PREFIX/current.new" "$PREFIX/current"
# the unit of the release we switch to: install.sh replaced it with the newer release's
install -m 0644 "$TARGET/deploy/$SERVICE.service" "/etc/systemd/system/$SERVICE.service"
systemctl daemon-reload
echo "Active release: $NAME${CURRENT:+ (previous: $(basename "$CURRENT"))}"

echo "Restarting $SERVICE (the active training run pauses at its next checkpoint first)"
systemctl reset-failed "$SERVICE" 2>/dev/null || true  # a crash loop of the release we leave would block the start
systemctl restart "$SERVICE"

# wait for readiness where `check` says the server listens
LISTEN="$(printf '%s\n' "$CHECKED" | sed -n 's/.*listening: http:\/\/\(.*\):\([0-9][0-9]*\), Host header.*/\1 \2/p' | head -n 1)"
HOST="${LISTEN% *}"
PORT="${LISTEN##* }"
case "$HOST" in '*'|0.0.0.0|::|'[::]'|'') HOST=127.0.0.1 ;; *:*) HOST="[$HOST]" ;; esac
if [ -n "$LISTEN" ] && command -v curl >/dev/null; then
  for _ in $(seq 1 90); do
    if READY="$(curl -fsS --max-time 2 "http://$HOST:$PORT/readyz" 2>/dev/null)"; then
      echo "Ready: $READY"
      exit 0
    fi
    sleep 1
  done
  echo "Not ready after 90 s: journalctl -u $SERVICE -n 100" >&2
  exit 1
fi
echo "Check that it is ready: curl -s localhost:<port>/readyz"
