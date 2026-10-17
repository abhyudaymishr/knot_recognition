#!/bin/bash
# Double-click this file in Finder to put the annotator model on the KnotAssistant Space
# (https://huggingface.co/spaces/AbhyudayMishr/KnotAssistant).
# It refreshes huggingface_space/ from this repository (the trained UNetFormer, its strand-order head
# and the code they need), lists the files, asks before uploading, uploads them with your own
# Hugging Face login and then follows the Space's rebuild until the new version runs.
# The first time it asks for a Hugging Face token with write access
# (https://huggingface.co/settings/tokens); it is stored on this Mac, so later runs do not ask.
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}

if ! $PY -c "import huggingface_hub" 2>/dev/null; then
  echo "Installing huggingface_hub (the Hugging Face upload library)..."
  $PY -m pip install --quiet huggingface_hub 2>/dev/null \
    || $PY -m pip install --quiet --user huggingface_hub 2>/dev/null \
    || $PY -m pip install --quiet --user --break-system-packages huggingface_hub
fi

$PY scripts/upload_space.py "$@"
status=$?
echo
if [ $status -eq 0 ]; then echo "Done."; else echo "Stopped (code $status); see the lines above."; fi
read -r -p "Press Return to close this window. " _
