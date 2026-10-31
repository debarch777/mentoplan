#!/usr/bin/env bash
# Download the public 3D Slicer sample CBCTs and process them into two ready-to-plan demo cases.
# Processing runs one case at a time in a memory-capped child process (needs ~3.5 GB free RAM,
# about 3 minutes per case).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MP="${MENTOPLAN_BIN:-$ROOT/.venv/bin/mentoplan}"
"$ROOT/scripts/get_sample_data.sh"
RAW="$ROOT/data/raw/slicer"
"$MP" ingest "$RAW/PreDentalSurgery.gipl.gz" --name "Sample patient A (CBCT)"
"$MP" ingest "$RAW/DZ-CBCT.nrrd" --name "Sample patient B (CBCT)"
echo "Done. Start the planner with: scripts/server.sh start   (http://127.0.0.1:8000)"
