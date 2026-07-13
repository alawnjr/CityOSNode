#!/usr/bin/env bash
#
# setup_realsense_pi.sh — build the Intel RealSense SDK (librealsense +
# pyrealsense2) from source on a Raspberry Pi and install the Python module
# into this repo's venv.
#
# Why source: PyPI/piwheels have no aarch64 pyrealsense2 wheel, so the SDK must
# be compiled on the Pi. The RSUSB backend talks to the camera purely through
# libusb (no kernel patches), which is the supported path on Raspberry Pi OS /
# plain Debian. Expect the compile to take 1–2 hours on a Pi 4.
#
# Usage (on the Pi, from ~/CityOS):   bash setup_realsense_pi.sh
# Needs sudo for: apt deps, make install, udev rules.
#
set -euo pipefail

LRS_TAG="${LRS_TAG:-v2.58.2}"
SRC_DIR="$HOME/librealsense"
VENV_PY="$HOME/CityOS/.venv/bin/python"
JOBS="${JOBS:-3}"   # -j4 can OOM the 4GB Pi 4 during the heaviest C++ units

[ -x "$VENV_PY" ] || { echo "venv python not found at $VENV_PY"; exit 1; }
SITE_PACKAGES=$("$VENV_PY" -c 'import site; print(site.getsitepackages()[0])')

echo ">> installing build dependencies (apt)"
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  git cmake build-essential libssl-dev libusb-1.0-0-dev libudev-dev \
  pkg-config python3-dev

if [ ! -d "$SRC_DIR" ]; then
  echo ">> cloning librealsense $LRS_TAG"
  git clone --depth 1 --branch "$LRS_TAG" \
    https://github.com/IntelRealSense/librealsense.git "$SRC_DIR"
fi

echo ">> configuring (RSUSB backend, python bindings -> venv)"
mkdir -p "$SRC_DIR/build"
cd "$SRC_DIR/build"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE="$VENV_PY" \
  -DPython_EXECUTABLE="$VENV_PY" \
  -DPython3_EXECUTABLE="$VENV_PY" \
  -DPYTHON_INSTALL_DIR="$SITE_PACKAGES/pyrealsense2" \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_GLSL_EXTENSIONS=OFF

echo ">> building with -j$JOBS (this is the 1–2 hour part)"
make -j"$JOBS"

echo ">> installing (sudo)"
sudo make install
sudo ldconfig

echo ">> installing udev rules (camera access without root)"
sudo cp "$SRC_DIR/config/99-realsense-libusb.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger

echo ">> verifying the venv can import pyrealsense2"
if ! "$VENV_PY" -c 'import pyrealsense2 as rs; print("pyrealsense2 OK:", rs.__version__ if hasattr(rs, "__version__") else "imported")'; then
  # older CMake layouts drop the module next to the build outputs instead of
  # honoring PYTHON_INSTALL_DIR — copy it into the venv by hand.
  echo ">> import failed; copying built module into the venv"
  mkdir -p "$SITE_PACKAGES"
  find "$SRC_DIR/build" -name 'pyrealsense2*.so' -exec cp -v {} "$SITE_PACKAGES/" \;
  "$VENV_PY" -c 'import pyrealsense2 as rs; print("pyrealsense2 OK after copy")'
fi

echo ">> done. Sanity check with the camera plugged in:"
echo "   $VENV_PY -c 'import pyrealsense2 as rs; print([d.get_info(rs.camera_info.name) for d in rs.context().devices])'"
