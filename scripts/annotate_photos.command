#!/bin/bash
# Double-click in Finder (or run in Terminal) to open the photo annotator with the trained UNetFormer
# suggesting width, crossings with over/under, free ends and centre-lines for each photo.
# It starts scripts/annotator_server.py (everything runs on this Mac) and opens the page in your browser.
# Options go after the name, e.g.:  bash scripts/annotate_photos.command --seed 2
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
echo "Starting the annotator. Close this window (or press Ctrl-C) to stop it."
"$PY" scripts/annotator_server.py "$@"
read -r -p "Press Return to close this window. " _
