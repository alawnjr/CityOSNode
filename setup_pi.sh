#!/usr/bin/env bash
# Post-reimage setup for the CityOS / smartroom node.
# Re-run safe. Run as the node user (NOT root): bash setup_pi.sh
set -euo pipefail

# ---- config -----------------------------------------------------------------
# The nodes only PULL, and the repo is public, so use HTTPS origin — no SSH key
# needed and `git pull` works keyless. (Editing/pushing happens on the dev
# machine, never on a Pi.)
REPO_HTTPS="https://github.com/alawnjr/CityOSNode.git"
CLONE_DIR="$HOME/CityOS"
SVC="smartroom-video-page.service"

echo "==> user=$USER  home=$HOME  clone=$CLONE_DIR"

# ---- 1. system packages -----------------------------------------------------
echo "==> Installing system packages (sudo — passwordless on these nodes)"
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip ffmpeg v4l-utils

# ---- 2. clone (or update) the repo -----------------------------------------
if [ -d "$CLONE_DIR/.git" ]; then
  echo "==> Repo exists, setting HTTPS origin and pulling"
  git -C "$CLONE_DIR" remote set-url origin "$REPO_HTTPS"
  git -C "$CLONE_DIR" pull origin master
else
  echo "==> Cloning repo (HTTPS)"
  git clone "$REPO_HTTPS" "$CLONE_DIR"
fi
echo "==> remotes:"; git -C "$CLONE_DIR" remote -v

# ---- 3. python venv + deps --------------------------------------------------
# Video capture (capture.py) and the web page are stdlib-only and run on system
# python3 — the venv is only for the test/ sensor scripts (adafruit/spidev/numpy).
echo "==> Creating venv + installing requirements (for test/ scripts)"
python3 -m venv "$CLONE_DIR/.venv"
"$CLONE_DIR/.venv/bin/pip" install --upgrade pip
"$CLONE_DIR/.venv/bin/pip" install -r "$CLONE_DIR/requirements.txt"

# ---- 4. web-page systemd unit ----------------------------------------------
echo "==> Installing $SVC (sudo)"
sudo tee "/etc/systemd/system/$SVC" >/dev/null <<UNIT
[Unit]
Description=Smartroom local video page
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$CLONE_DIR
ExecStart=/usr/bin/python3 $CLONE_DIR/smartroom_video_page.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now "$SVC"

# ---- 5. disable wifi power-save (the likely cause of the earlier drops) -----
echo "==> Disabling wifi power save (persistent)"
sudo tee /etc/systemd/system/wifi-powersave-off.service >/dev/null <<'UNIT'
[Unit]
Description=Disable wlan0 power save
After=network.target
[Service]
Type=oneshot
ExecStart=/usr/sbin/iw dev wlan0 set power_save off
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl enable --now wifi-powersave-off.service || true

# ---- 6. done ----------------------------------------------------------------
cat <<EOF

==> DONE on $(hostname). No SSH key / sudo password needed (public repo, pull-only,
    passwordless sudo). Optional: recreate PRIVATE.md (credential notes only).

Quick checks:
  v4l2-ctl --list-devices                                  # camera shows up?
  python3 $CLONE_DIR/capture.py -d 3                        # records a 3s clip
  systemctl status $SVC                                     # web page running
  curl -sI http://localhost:8000 | head -1                 # web page responding
EOF
