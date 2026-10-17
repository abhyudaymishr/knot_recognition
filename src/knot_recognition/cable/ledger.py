"""
ledger.py - "Score once" bookkeeping for held-out real-image sets.

A held-out set (real cable photographs, the blind real-picture set) is frozen by the
SHA-256 of its manifest. Every scoring run appends a line to results/heldout_ledger.jsonl:
set name and hash, model role, checkpoint name and hash, time, git commit.

Rule: for role="main", only ONE checkpoint may ever be scored on a frozen set. A second,
different main checkpoint is refused unless `allow_new_main` is given with a reason, and
that reason is recorded (and must be reported in the paper). Baselines and ablations
(role != "main") may be scored freely: they are not tuned on the set either, but they do
not change the headline model.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "results" / "heldout_ledger.jsonl"
FROZEN = ROOT / "results" / "frozen_model.json"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_checkpoint() -> dict:
    return json.load(open(FROZEN))


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa
        return None


def record_run(set_name: str, set_hash: str, ckpt_path, role: str = "main",
               allow_new_main: Optional[str] = None, note: str = "") -> dict:
    ck_hash = sha256_file(ckpt_path) if ckpt_path and Path(ckpt_path).exists() else None
    runs = [json.loads(l) for l in open(LEDGER)] if LEDGER.exists() else []
    prior_main = {r["checkpoint_sha256"] for r in runs
                  if r["set_hash"] == set_hash and r["role"] == "main"}
    if role == "main":
        fz = frozen_checkpoint()
        if ck_hash != fz["sha256"] and not allow_new_main:
            raise SystemExit(f"refused: {Path(ckpt_path).name} is not the frozen model "
                             f"({fz['checkpoint']}, sha256 {fz['sha256'][:12]}). Pass a reason to score another main model.")
        if prior_main and ck_hash not in prior_main and not allow_new_main:
            raise SystemExit(f"refused: set {set_name} was already scored with another main model; "
                             "scoring a new one requires a recorded reason")
    entry = {"set": set_name, "set_hash": set_hash, "role": role,
             "checkpoint": Path(ckpt_path).name if ckpt_path else None, "checkpoint_sha256": ck_hash,
             "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": _git_commit(),
             "repeat_of_same_model": ck_hash in prior_main, "new_main_reason": allow_new_main, "note": note}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry
