#!/usr/bin/env bash
# Apply /etc/yolo-workbench/config.toml to the system: create the projects folder and the cache it names when
# they are missing (owned by the service account), and write the systemd unit for them: the folders the
# service may write and waits for, and its stop timeout. Run as root after changing projects_root, cache_dir,
# pause_deadline_s or shutdown_deadline_s (install.sh and rollback.sh run it too), then restart:
#
#   sudo bash /opt/yolo-workbench/current/deploy/apply-config.sh
#   sudo systemctl restart yolo-workbench
#
# The unit comes from `yolo-workbench-server unit` of the release given as the argument (default: current);
# do not edit /etc/systemd/system/yolo-workbench.service by hand, the next run of this script replaces it.
set -euo pipefail

PREFIX=/opt/yolo-workbench
ACCOUNT=yolo-workbench
CONFIG=/etc/yolo-workbench/config.toml
ENV_FILE=/etc/yolo-workbench/yolo-workbench.env
SERVICE=yolo-workbench
UNIT_FILE="/etc/systemd/system/$SERVICE.service"
RELEASE="${1:-$PREFIX/current}"
SERVER="$RELEASE/.venv/bin/yolo-workbench-server"

if [ "$(id -u)" -ne 0 ]; then echo "Run as root (sudo bash $0)" >&2; exit 1; fi
if [ ! -x "$SERVER" ]; then echo "$RELEASE is not an installed release (no $SERVER)" >&2; exit 1; fi
if [ ! -f "$CONFIG" ]; then echo "$CONFIG does not exist; install.sh creates it from deploy/config.example.toml" >&2; exit 1; fi
if ! id -u "$ACCOUNT" >/dev/null 2>&1; then echo "There is no $ACCOUNT account; install.sh creates it" >&2; exit 1; fi
cd /

render() { "$SERVER" unit --config "$CONFIG" --env-file "$ENV_FILE" --account "$ACCOUNT" --prefix "$PREFIX" "$@"; }
FOLDERS="$(render --folders)"  # the projects folder first, then the cache
UNIT="$(render)"

# whether the installed unit already lets the service write $1 (paths have no spaces: `unit` refuses them)
in_installed_unit() {
  local path
  for path in $(systemctl show "$SERVICE" --property=ReadWritePaths --value 2>/dev/null || true); do
    path="${path#-}"
    if [ "${path#+}" = "$1" ]; then return 0; fi
  done
  return 1
}

PROJECTS=1
while IFS= read -r folder; do
  if [ ! -d "$folder" ]; then
    # a projects folder the service already used is missing when its disk is not mounted: creating it would
    # put an empty workspace on the system disk (the reason for create_projects_root = false)
    if [ "$PROJECTS" -eq 1 ] && in_installed_unit "$folder"; then
      echo "The projects folder $folder is missing, although the service already uses it: is its disk mounted?" >&2
      echo "Mount it (or for a new, empty one: sudo install -d -o $ACCOUNT -g $ACCOUNT -m 0750 $folder) and run this again." >&2
      exit 1
    fi
    install -d -o "$ACCOUNT" -g "$ACCOUNT" -m 0750 "$folder"
    echo "Created $folder (owner $ACCOUNT)"
  elif ! runuser -u "$ACCOUNT" -- test -w "$folder"; then
    echo "Warning: $ACCOUNT cannot write $folder (owner $(stat -c %U "$folder"), mode $(stat -c %a "$folder")): sudo chown $ACCOUNT: $folder" >&2
  fi
  PROJECTS=0
done <<< "$FOLDERS"

if [ -f "$UNIT_FILE" ] && [ "$(cat "$UNIT_FILE")" = "$UNIT" ]; then
  echo "$UNIT_FILE is up to date"
else
  printf '%s\n' "$UNIT" > "$UNIT_FILE.new"
  chmod 0644 "$UNIT_FILE.new"
  mv -f "$UNIT_FILE.new" "$UNIT_FILE"
  systemctl daemon-reload
  echo "Wrote $UNIT_FILE for $CONFIG (the service may write: ${FOLDERS//$'\n'/ })"
  if systemctl is-active --quiet "$SERVICE"; then
    echo "The running service still uses the previous unit: sudo systemctl restart $SERVICE"
  fi
fi

# drop-ins change the unit too; one from an older guide may still add folders or a stop timeout by hand
DROPINS="$(systemctl show "$SERVICE" --property=DropInPaths --value 2>/dev/null || true)"
if [ -n "$DROPINS" ]; then
  echo "Note: these drop-ins also change the unit: $DROPINS"
  echo "      the folders and TimeoutStopSec come from $CONFIG now; remove those settings from them."
fi
