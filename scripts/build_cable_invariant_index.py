"""Precompute a small lookup table for the bundled knot braid census."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_knotgen(path: Path):
    spec = importlib.util.spec_from_file_location("cable_index_knotgen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Knotgen source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(census_path: Path, knotgen_path: Path, output_path: Path) -> None:
    kg = load_knotgen(knotgen_path)
    census = json.loads(census_path.read_text())
    index = {}
    for label, row in census.items():
        polynomial = kg.alexander_burau(row["braid"], row["nstrands"])
        signature = (
            str(polynomial),
            abs(int(polynomial.subs(kg.t, -1))),
            int(row["crossings"]),
        )
        index.setdefault(signature, []).append(label)
    records = [
        {"alexander_polynomial": sig[0], "determinant": sig[1], "crossings": sig[2], "labels": labels}
        for sig, labels in sorted(index.items(), key=lambda item: (item[0][2], item[0][0], item[0][1]))
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Wrote {len(records)} invariant fingerprints for {len(census)} census entries to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, default=Path("Knotgen/census_braids.json"))
    parser.add_argument("--knotgen", type=Path, default=Path("Knotgen/knotgen.py"))
    parser.add_argument("--out", type=Path, default=Path("huggingface_space/runtime/Knotgen/census_invariant_index.json"))
    args = parser.parse_args()
    build(args.census, args.knotgen, args.out)


if __name__ == "__main__":
    main()
