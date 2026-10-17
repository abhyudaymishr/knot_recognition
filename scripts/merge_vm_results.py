"""merge_vm_results.py - Merge a results tar made on the Mac Linux VM into this repo (new ledger lines appended with a note, files copied).
Usage: python scripts/merge_vm_results.py RESULTS.tar"""
import json, shutil, sys, tarfile, tempfile
from pathlib import Path
repo = Path(__file__).resolve().parent.parent
tar = Path(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    tarfile.open(tar).extractall(td)
    T = Path(td)
    led = T / "results" / "heldout_ledger.jsonl"
    if led.exists():
        cloud = [json.loads(l) for l in open(repo / "results/heldout_ledger.jsonl")]
        key = lambda e: (e.get("set"), e.get("role"), e.get("checkpoint_sha256"), e.get("time_utc"))
        have = {key(e) for e in cloud}
        new = [json.loads(l) for l in open(led) if key(json.loads(l)) not in have]
        with open(repo / "results/heldout_ledger.jsonl", "a") as f:
            for e in new:
                e["scored_on"] = "Mac Linux VM (Claude desktop), copied to this ledger"
                f.write(json.dumps(e) + "\n")
        print("ledger: +", len(new), [(e["set"], e["role"], e["note"][:60]) for e in new])
        led.unlink()
    n = 0
    for p in T.rglob("*"):
        if p.is_file():
            dst = repo / p.relative_to(T)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            n += 1
    print("files copied:", n)
