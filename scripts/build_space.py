"""
build_space.py - Refresh huggingface_space/ (the KnotAssistant Space) from this repository.

Copies the modules the Space imports, the trained UNetFormer and its strand-order head, Knotgen
and the KnotInfo table into huggingface_space/runtime/ (and three synthetic example photos into
huggingface_space/examples/), and writes runtime/MANIFEST.json with the
source and sha256 of every copied file (and the repository commit) so the Space says exactly
which model it runs. app.py, README.md and requirements.txt are edited by hand and not touched.

  python scripts/build_space.py              # UNetFormer seed 0 + its strand-order head
  python scripts/build_space.py --seed 2
  python scripts/build_space.py --check      # only report what is out of date (exit 1 if anything)
  python scripts/build_space.py --skip-missing   # keep the Space's copy of files missing here
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "huggingface_space"
RT = SPACE / "runtime"

MODULES = ["utils", "preprocess", "gauss_pd", "nanotransformer", "traverser",             # tab 1
           "knot_aware", "unetformer", "strand_pointer", "minkowski", "topology_loss", "annotate"]
CABLE = ["consistency", "denoise", "diagram", "from_traversal", "gap_measure", "identify", "moves",
         "neural_parser", "uncertainty"]
INIT = '"""Runtime copy of the knot_recognition package for the Space (scripts/build_space.py)."""\n'


def files(seed: int):
    src = ROOT / "src" / "knot_recognition"
    out = {f"knot_recognition/{m}.py": src / f"{m}.py" for m in MODULES}
    out["knot_recognition/cable/__init__.py"] = src / "cable" / "__init__.py"
    out.update({f"knot_recognition/cable/{m}.py": src / "cable" / f"{m}.py" for m in CABLE})
    out.update({f"Knotgen/{f}": ROOT / "Knotgen" / f for f in ("knotgen.py", "census_braids.json")})
    out["knotinfo_full_dataset.json"] = ROOT / "data" / "knotinfo" / "knotinfo_full_dataset.json"
    out["nanotransformer_knot250.pth"] = ROOT / "checkpoints" / "nanotransformer_knot250.pth"
    out[f"mps_uf_s{seed}.pth"] = ROOT / "checkpoints" / "beat_unet" / f"mps_uf_s{seed}.pth"
    out[f"mps_ptr_s{seed}.pth"] = ROOT / "checkpoints" / "strand_pointer" / f"mps_ptr_s{seed}.pth"
    return out


# example photos for the annotation tab: synthetic KnotBench-Photo renders the model reads correctly
EXAMPLES = {"1_unknot_c001_S2_desk.jpg": "c001_S2_desk", "2_trefoil_c007_S2_desk.jpg": "c007_S2_desk",
            "3_figure_eight_c019_S4_clutter.jpg": "c019_S4_clutter"}


def example_files():
    img = ROOT / "data" / "knotbench_photo" / "synthetic_v1" / "images"
    return {name: img / f"{iid}.jpg" for name, iid in EXAMPLES.items()}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="which trained UNetFormer and strand-order head")
    ap.add_argument("--check", action="store_true", help="only report out-of-date files")
    ap.add_argument("--skip-missing", action="store_true",
                    help="keep the Space's copy of a file that is missing in the repository instead of stopping")
    a = ap.parse_args()

    # (copy in the Space, source in the repository)
    pairs = [(RT / rel, src) for rel, src in files(a.seed).items()]
    pairs += [(SPACE / "examples" / name, src) for name, src in example_files().items()]
    stale, missing, kept = [], [], []
    for dst, src in pairs:
        shown = str(dst.relative_to(SPACE))
        if not src.exists():
            (kept if a.skip_missing and dst.exists() else missing).append(shown)
        elif not dst.exists() or sha(dst) != sha(src):
            stale.append(shown)
            if not a.check:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    if missing:
        sys.exit("missing in the repository (and not in the Space folder either): " + ", ".join(missing))
    if kept:
        print("kept the Space folder's copy of files missing in the repository: " + ", ".join(kept))
    init = RT / "knot_recognition" / "__init__.py"
    if not init.exists() or init.read_text() != INIT:
        stale.append("runtime/knot_recognition/__init__.py")
        if not a.check:
            init.parent.mkdir(parents=True, exist_ok=True)
            init.write_text(INIT)

    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:  # noqa
        commit = None
    manifest = {"model": f"mps_uf_s{a.seed}.pth", "strand_head": f"mps_ptr_s{a.seed}.pth", "kappa": 26.5,
                "repository_commit": commit,
                "files": {rel: {"source": str(src.relative_to(ROOT)), "sha256": sha(src if src.exists() else RT / rel)[:12]}
                          for rel, src in sorted(files(a.seed).items())}}
    mf = RT / "MANIFEST.json"
    old = json.loads(mf.read_text()) if mf.exists() else {}
    same = {k: v for k, v in old.items() if k != "repository_commit"} == {k: v for k, v in manifest.items() if k != "repository_commit"}
    if not same:
        stale.append("runtime/MANIFEST.json")
        if not a.check:
            mf.write_text(json.dumps(manifest, indent=1) + "\n")

    if a.check:
        print("up to date" if not stale else "out of date: " + ", ".join(stale))
        sys.exit(1 if stale else 0)
    print(f"huggingface_space: {len(stale)} file(s) refreshed" + (": " + ", ".join(stale) if stale else ""))


if __name__ == "__main__":
    main()
