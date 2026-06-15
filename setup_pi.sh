#!/usr/bin/env bash
# Post-reimage setup for the CityOS / smartroom node.
# Re-run safe. Run as the node user (NOT root): bash setup_pi.sh
set -euo pipefail

# ---- config -----------------------------------------------------------------
REPO_SSH="git@github.com:alawnjr/CityOSNode.git"   # origin (Pi pulls from here)
REPO_HTTPS="https://github.com/alawnjr/CityOSNode.git"
PERSONAL_SSH="git@gitlab.orbit-lab.org:alawnjr/smartroom.git"
CLONE_DIR="$HOME/CityOS"
SVC="smartroom-video-page.service"

echo "==> user=$USER  home=$HOME  clone=$CLONE_DIR"

# ---- 1. system packages -----------------------------------------------------
echo "==> Installing system packages (needs sudo)"
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip ffmpeg v4l-utils

# ---- 2. SSH key for git (so the Pi can pull/push) ---------------------------
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
  echo "==> Generating SSH key"
  ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519"
fi
echo
echo "    >>> Add this public key to GitHub (and GitLab) deploy keys, then press Enter:"
echo "    ----------------------------------------------------------------------"
cat "$HOME/.ssh/id_ed25519.pub"
echo "    ----------------------------------------------------------------------"
read -r _ </dev/tty

# ---- 3. clone (or update) the repo -----------------------------------------
if [ -d "$CLONE_DIR/.git" ]; then
  echo "==> Repo exists, pulling"
  git -C "$CLONE_DIR" pull origin master
else
  echo "==> Cloning repo"
  # try SSH first (so push works); fall back to HTTPS read-only clone
  git clone "$REPO_SSH" "$CLONE_DIR" || git clone "$REPO_HTTPS" "$CLONE_DIR"
fi

# ---- 4. git remotes (origin=GitHub, personal=GitLab; push to both) ----------
git -C "$CLONE_DIR" remote set-url origin "$REPO_SSH" 2>/dev/null \
  || git -C "$CLONE_DIR" remote add origin "$REPO_SSH"
git -C "$CLONE_DIR" remote get-url personal >/dev/null 2>&1 \
  || git -C "$CLONE_DIR" remote add personal "$PERSONAL_SSH"
echo "==> remotes:"; git -C "$CLONE_DIR" remote -v

# ---- 5. python venv + deps --------------------------------------------------
echo "==> Creating venv + installing requirements"
python3 -m venv "$CLONE_DIR/.venv"
"$CLONE_DIR/.venv/bin/pip" install --upgrade pip
"$CLONE_DIR/.venv/bin/pip" install -r "$CLONE_DIR/requirements.txt"

# ---- 6. web-page systemd unit ----------------------------------------------
echo "==> Installing $SVC (needs sudo)"
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

# ---- 7. disable wifi power-save (the likely cause of the earlier drops) -----
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

# ---- 8. reminders -----------------------------------------------------------
cat <<EOF

==> DONE. Manual steps left:
  1. Recreate PRIVATE.md in $CLONE_DIR (Pi credentials — gitignored, not in repo).
  2. Confirm SSH key was authorized on GitHub & GitLab (push test below).
  3. Re-clone smartroom-autolabeling separately if you use it.

Quick checks:
  v4l2-ctl --list-devices                 # camera shows up?
  $CLONE_DIR/.venv/bin/python $CLONE_DIR/check_av.py   # camera + mics
  systemctl status $SVC                    # web page running
  curl -sI http://localhost:8000 | head -1 # web page responding
EOF
