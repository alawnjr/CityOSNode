#!/bin/sh
# upload_recording.sh — push finished recordings from this node to the quad
# server's data store. Runs ON THE PI (smartroom2), driven by the
# smartroom-upload.timer/.service systemd pair (or by hand).
#
# A recording is "finished" once capture.py has written its metadata.json (it
# writes that LAST). This script scans data/ for every rec_* dir that has a
# metadata.json but no .uploaded marker, rsyncs each to the server, and drops
# the marker on success. Idempotent and self-healing: a failed or interrupted
# upload leaves no marker, so the next run retries it; already-uploaded clips
# are skipped. rsync itself only sends changed/new bytes.
#
#   ./upload_recording.sh                 # scan + upload all pending recordings
#   ./upload_recording.sh data/day_../rec_..   # upload one specific rec dir
#
# Config (from the environment; the systemd service loads node.env via
# EnvironmentFile so it can be pinned per node there):
#   SMARTROOM_UPLOAD_DEST   user@host:/abs/recordings/root on the server
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DATA_DIR="$SCRIPT_DIR/data"

DEST="${SMARTROOM_UPLOAD_DEST:-intern26@172.16.60.239:/mnt/data4/intern26/recordings}"
# This node's id in the merged session tree (streams/<node_id>/). smartroom2 is
# "cam2" (see smartroom-control lib/nodes.ts / the mirror's CAM_NAME map).
NODE_ID="${SMARTROOM_NODE_ID:-cam2}"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"

# rsync one rec dir, preserving the day_NN/rec_NNN suffix under DEST. Returns
# rsync's exit status so the caller can decide whether to mark it done.
upload_one() {
  rec_dir=$1
  [ -f "$rec_dir/metadata.json" ] || { echo "skip (no metadata.json): $rec_dir" >&2; return 1; }
  [ -d "$rec_dir/streams" ] || { echo "skip (no streams/): $rec_dir" >&2; return 1; }
  # rel = day_NN_DATE/rec_DATE_NNN  (path relative to data/)
  rel=${rec_dir#"$DATA_DIR"/}
  dest_host=${DEST%%:*}
  dest_root=${DEST#*:}
  # Restructure into the layout the analysis + mirror expect (what the laptop's
  # "Save All" used to produce): this node's clips go UNDER streams/<node_id>/,
  # with a copy of the rec's metadata.json alongside them — localize.py reads
  # metadata via mp4.parent/metadata.json, so it must sit next to the mp4s.
  #   /mnt/data4/.../recordings/<day>/<rec>/streams/<node_id>/{metadata.json, camera_*.mp4, ...}
  cam_dir="$dest_root/$rel/streams/$NODE_ID"
  ssh $SSH_OPTS "$dest_host" "mkdir -p '$cam_dir'" || return $?
  rsync -a --partial -e "ssh $SSH_OPTS" "$rec_dir/streams/" "$dest_host:$cam_dir/" || return $?
  rsync -a --partial -e "ssh $SSH_OPTS" "$rec_dir/metadata.json" "$dest_host:$cam_dir/metadata.json" || return $?
  # Touch a sentinel at the recordings ROOT so the server's analyze .path unit
  # fires even when the clip landed inside an existing day_NN/ dir (which would
  # otherwise not change the watched root's mtime).
  ssh $SSH_OPTS "$dest_host" "touch '$dest_root/.last_upload'" || true
}

# Explicit single-dir mode.
if [ "$#" -gt 0 ]; then
  rc=0
  for d in "$@"; do
    d=${d%/}
    if upload_one "$d"; then
      : > "$d/.uploaded"
      echo "uploaded: $d" >&2
    else
      rc=1
      echo "FAILED: $d (will retry next run)" >&2
    fi
  done
  exit "$rc"
fi

# Scan mode: every finished, not-yet-uploaded recording.
[ -d "$DATA_DIR" ] || { echo "no data dir: $DATA_DIR" >&2; exit 0; }
rc=0
found=0
for meta in "$DATA_DIR"/*/rec_*/metadata.json; do
  [ -f "$meta" ] || continue          # glob didn't match -> literal, skip
  rec_dir=${meta%/metadata.json}
  [ -f "$rec_dir/.uploaded" ] && continue
  found=1
  if upload_one "$rec_dir"; then
    : > "$rec_dir/.uploaded"
    echo "uploaded: $rec_dir" >&2
  else
    rc=1
    echo "FAILED: $rec_dir (will retry next run)" >&2
  fi
done
[ "$found" = 0 ] && echo "nothing pending" >&2
exit "$rc"
