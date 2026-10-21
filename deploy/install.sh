#!/usr/bin/env bash
# Install or upgrade the YOLO Workbench server from this source checkout. Run as root on the server:
#
#   sudo deploy/install.sh               # version name from `git describe`
#   sudo deploy/install.sh 2026.10.21    # or name it
#
# Each version goes into its own folder (/opt/yolo-workbench/releases/<version>/, code + virtual environment +
# release.json, which the server reports); /opt/yolo-workbench/current points at the active one and
# /opt/yolo-workbench/previous at the one before. The running server keeps using its own release's files
# until `systemctl restart`. Older releases are kept: deploy/rollback.sh switches back to one.
# The projects folder, the cache and the systemd unit follow /etc/yolo-workbench/config.toml (deploy/apply-config.sh;
# the first install creates it from deploy/config.example.toml, so write it beforehand for other folders).
# Needs uv (https://docs.astral.sh/uv/) on the PATH and internet access for the first install.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-$(git -C "$SRC" describe --tags --always --dirty 2>/dev/null || date +%Y%m%d-%H%M)}"
PREFIX=/opt/yolo-workbench
RELEASE="$PREFIX/releases/$VERSION"
ACCOUNT=yolo-workbench
CONFIG=/etc/yolo-workbench/config.toml
UV="${UV:-$(command -v uv || true)}"

if [ "$(id -u)" -ne 0 ]; then echo "Run as root (sudo $0)" >&2; exit 1; fi
if [ -z "$UV" ]; then echo "uv is required: curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin sh" >&2; exit 1; fi
if [[ ! "$VERSION" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then echo "Invalid version name '$VERSION' (letters, digits, . _ + -)" >&2; exit 1; fi
if [ -e "$RELEASE" ]; then echo "$RELEASE already exists; pass another version name" >&2; exit 1; fi
case "$VERSION" in *-dirty) echo "Warning: the checkout has uncommitted changes; they (and untracked files) are installed too" >&2 ;; esac

# the service account owns only the projects folder and the cache (apply-config.sh); the code stays root-owned
# and read-only. It needs no home folder: the server points the libraries' caches and HOME into the cache.
id -u "$ACCOUNT" >/dev/null 2>&1 || useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin "$ACCOUNT"
for group in video render; do
  if getent group "$group" >/dev/null; then usermod -aG "$group" "$ACCOUNT"; fi  # GPU device nodes on some distributions
done
install -d -m 0755 "$PREFIX/releases" /etc/yolo-workbench

echo "Installing $VERSION into $RELEASE"
mkdir -p "$RELEASE"
# a failed install leaves no half-built release behind (its name could not be used again)
trap 'echo "Install failed; removing the incomplete $RELEASE" >&2; rm -rf "$RELEASE"' ERR
tar -C "$SRC" --exclude=./.git --exclude=./.venv --exclude='__pycache__' --exclude=./.pytest_cache -cf - . | tar -C "$RELEASE" -xf -
# what the server reports as its release (/readyz, /api/v1/app, `check`, the log); the name is validated above
COMMIT="$(git -C "$SRC" rev-parse HEAD 2>/dev/null || true)"
printf '{"name": "%s", "commit": "%s", "installed": "%s"}\n' "$VERSION" "$COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RELEASE/release.json"
# Python and the environment live under /opt, not in root's home (the service cannot read /root); always
# uv's own Python, even when the system has a 3.12 (uv would pick that one and leave $PREFIX/python empty)
UV_PYTHON_INSTALL_DIR="$PREFIX/python" UV_CACHE_DIR="$PREFIX/.uv-cache" UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$RELEASE/.venv" \
  "$UV" sync --project "$RELEASE" --frozen --no-dev --python 3.12 --managed-python
chmod -R a+rX "$RELEASE" "$PREFIX/python"

# configuration: installed once, never overwritten
CREATED=0
if [ ! -e "$CONFIG" ]; then
  install -m 0644 -g "$ACCOUNT" "$RELEASE/deploy/config.example.toml" "$CONFIG"
  CREATED=1
fi
if [ ! -e /etc/yolo-workbench/yolo-workbench.env ]; then
  install -m 0644 -g "$ACCOUNT" "$RELEASE/deploy/yolo-workbench.env" /etc/yolo-workbench/yolo-workbench.env
fi
# the folders and the systemd unit for the configuration, from this release's `yolo-workbench-server unit`
bash "$RELEASE/deploy/apply-config.sh" "$RELEASE"

PREVIOUS="$(readlink -f "$PREFIX/current" 2>/dev/null || true)"
if [ -n "$PREVIOUS" ]; then
  ln -sfn "$PREVIOUS" "$PREFIX/previous.new"
  mv -T "$PREFIX/previous.new" "$PREFIX/previous"
fi
ln -sfn "$RELEASE" "$PREFIX/current.new"
mv -T "$PREFIX/current.new" "$PREFIX/current"
trap - ERR

echo
echo "Installed $VERSION.${PREVIOUS:+ Previous release: $(basename "$PREVIOUS") (rollback: sudo bash $RELEASE/deploy/rollback.sh)}"
if [ "$CREATED" -eq 1 ]; then
  echo "Created $CONFIG from the example: set allowed_hosts before starting. To use other folders than"
  echo "         the example's, set projects_root / cache_dir there, then: sudo bash $PREFIX/current/deploy/apply-config.sh"
fi
echo "Account: sudo -u $ACCOUNT $PREFIX/current/.venv/bin/yolo-workbench-server user add <name> --admin --config $CONFIG"
echo "Check:   sudo -u $ACCOUNT $PREFIX/current/.venv/bin/yolo-workbench-server check --config $CONFIG"
echo "Start:   sudo systemctl enable --now yolo-workbench    (after an upgrade: Check, then sudo systemctl restart yolo-workbench)"
echo "Logs:    journalctl -u yolo-workbench -f"
