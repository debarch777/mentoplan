#!/usr/bin/env bash
# Download the 3D Slicer sample CBCT scans used for development and validation (SHA-256 verified).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/data/raw/slicer"; mkdir -p "$DST"; cd "$DST"
U=https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/
get() { [[ -f "$2" ]] || curl -sSL -o "$2" "${U}$1"; echo "$1  $2" | sha256sum -c -; }
get 7bfa16945629c319a439f414cfb7edddd2a97ba97753e12eede3b56a0eb09968 PreDentalSurgery.gipl.gz
get 4cdc3dc35519bb57daeef4e5df89c00849750e778809e94971d3876f95cc7bbd PostDentalSurgery.gipl.gz
get 4ce7aa75278b5a7b757ed0c8d7a6b3caccfc3e2973b020532456dbc8f3def7db DZ-CBCT.nrrd
echo "sample data in $DST"
