#!/usr/bin/env bash
# Add your site-specific #SBATCH header above the commands below.
set -euo pipefail

# Activate the Python environment with the packages in requirements.txt here.
CONFIG=${CONFIG:-config.yaml}
START=${START:?Set START to the inclusive start date, YYYY-MM-DD}
END=${END:?Set END to the exclusive end date, YYYY-MM-DD}

python -m brdf --config "$CONFIG" --start "$START" --end "$END"
