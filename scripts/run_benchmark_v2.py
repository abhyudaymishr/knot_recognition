"""
run_benchmark_v2.py - KnotBench v2: the paper's headline protocol.

Parts (run individually or all):
  features  Cache diagram-only features and hyperbolic invariants for all 2,978 knots.
  task2     Headline invariant prediction: train c<=10, checkpoint on c=11, test c=12.
            Diagram-only vs oracle-feature tabular models, four matched graph backbones.
  curve     Learning curve: train c<=9, c<=10, c<=10 + 70% of c=11 (fixed 30% c=11 val).
  davies    Reproduction of Davies et al. (2021): signature from hyperbolic invariants,
            on the leakage-free crossing split and on their random-split setting.
  task1     Hard-negative distinct-presentation verification with untrained,
            supervised-ID and contrastive encoders for every backbone; exterior
            isometry reported as a certificate ceiling with per-pair runtime.
  gain      Paired statistics for the learned gain in the c<=8 -> c=9 protocol
            (trained canonical GNN vs untrained canonical GNN, +0.008 AUROC).
  stats     Paired knot-level bootstrap tests (Holm-corrected) over all saved predictions.

Outputs: results/benchmark_v2/<part>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import snappy
import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from knot_recognition.bench_backbones import (
    BACKBONES, EmbeddingModel, RegressionModel, count_params,
)
from run_rigorous_tasks import (
    build_pure_graph_tensors, canonicalize_pd_snappy, compute_wl_graph_hash,
    extract_diagram_geometric_summary, is_knot_alternating, parse_braid_word,
    safe_randomize_pd,
)

OUT_DIR = ROOT / "results" / "benchmark_v2"
DB_PATH = ROOT / "data" / "knotinfo" / "knotinfo_full_dataset.json"
FEAT_CACHE = ROOT / "data" / "benchmark_v2_features.json"
SEEDS = [42, 43, 44, 45, 46]
TARGETS = ["unknotting_number", "smooth_four_genus", "three_genus", "signature"]
TARGET_PARAMS = 110_000   # parameter budget every backbone is matched to
EPOCHS = 60
BATCH = 32
LR = 1e-3

torch.set_num_threads(1)


# =============================================================================
# Utilities
# =============================================================================

def load_db() -> Dict[str, Any]:
    with open(DB_PATH) as f:
        return json.load(f)


def save(part: str, obj: Any) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{part}.json"
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] {p}")
    return p


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def matched_hidden(backbone: str, kind: str = "reg", target: int = TARGET_PARAMS) -> int:
    """Hidden width whose parameter count is closest to the shared budget."""
    best, best_gap = 64, float("inf")
    for h in range(24, 161, 4):
        m = RegressionModel(backbone, 4, hidden=h) if kind == "reg" else EmbeddingModel(backbone, hidden=h, num_classes=0)
        gap = abs(count_params(m) - target)
        if gap < best_gap:
            best, best_gap = h, gap
    return best


def split_by_crossings(db, lo, hi) -> List[str]:
    return [k for k, v in db.items() if lo <= v["crossings"] <= hi]


# =============================================================================
# Features
# =============================================================================

def hyperbolic_invariants(pd) -> Optional[Dict[str, float]]:
    """Davies et al. (2021) style invariants of the knot complement (None if not hyperbolic)."""
    try:
        M = snappy.Link(list(pd)).exterior()
        if M.solution_type() != "all tetrahedra positively oriented":
            M.randomize()
            if M.solution_type() != "all tetrahedra positively oriented":
                return None
        vol = float(M.volume())
        if vol < 1e-6:
            return None
        cs = float(M.chern_simons())
        (mu, lam), = M.cusp_translations()
        mu = complex(mu)
        lam = complex(lam)
        shape = complex(M.cusp_info()[0]["shape"])
        # natural slope Re(lambda/mu) (Davies et al. 2021, Eq. 1)
        slope = (lam / mu).real
        return {
            "volume": vol, "chern_simons": cs,
            "mu_re": mu.real, "mu_im": mu.imag, "lam_re": lam.real, "lam_im": lam.imag,
            "shape_re": shape.real, "shape_im": shape.imag,
            "cusp_volume": abs((mu.conjugate() * lam).imag),
            "slope": slope,
        }
    except Exception:
        return None


def part_features(db) -> Dict[str, Any]:
    if FEAT_CACHE.exists():
        print(f"[cache] {FEAT_CACHE}")
        with open(FEAT_CACHE) as f:
            return json.load(f)
    out = {}
    t0 = time.time()
    for i, (k, v) in enumerate(db.items()):
        can = canonicalize_pd_snappy(v["pd"]) if v["pd"] else v["pd"]
        geo = extract_diagram_geometric_summary(can).tolist() if can else [0.0] * 13
        out[k] = {"diagram": geo, "hyperbolic": hyperbolic_invariants(v["pd"]) if v["pd"] else None}
        if (i + 1) % 250 == 0:
            print(f"  features {i+1}/{len(db)} ({time.time()-t0:.0f}s)")
    FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEAT_CACHE, "w") as f:
        json.dump(out, f)
    return out


DIAGRAM_COLS = ["c", "n_pos", "n_neg", "writhe"] + [f"faces_deg{k}" for k in range(2, 11)]


def diagram_matrix(feats, ks) -> np.ndarray:
    return np.array([feats[k]["diagram"] for k in ks], dtype=np.float64)


def oracle_matrix(db, feats, ks, target: str) -> Tuple[np.ndarray, List[str]]:
    """Diagram features + classical algebraic invariants, minus any column equal to the target."""
    cols = DIAGRAM_COLS + ["signature", "abs_sig_half", "determinant", "alex_deg_half"] + [f"alex_c{i}" for i in range(8)]
    rows = []
    for k in ks:
        v = db[k]
        sig = float(v.get("signature") or 0.0)
        det = float(v.get("determinant") or 1.0)
        av = v.get("alexander_vector") or []
        deg, coeffs = 0.0, [0.0] * 8
        if isinstance(av, list) and len(av) > 2:
            deg = (av[1] - av[0]) / 2.0
            for i, c in enumerate(av[2:10]):
                coeffs[i] = float(c)
        rows.append(list(feats[k]["diagram"]) + [sig, abs(sig) / 2.0, det, deg] + coeffs)
    X = np.array(rows, dtype=np.float64)
    if target == "signature":
        keep = [i for i, c in enumerate(cols) if c not in ("signature", "abs_sig_half")]
        X, cols = X[:, keep], [cols[i] for i in keep]
    return X, cols


def targets_of(db, ks, tgt) -> Tuple[np.ndarray, np.ndarray]:
    y = np.array([float(db[k][tgt]) if db[k][tgt] is not None else np.nan for k in ks])
    return y, ~np.isnan(y)


# =============================================================================
# Graph regression training (per-target checkpoint selection on validation)
# =============================================================================

_GRAPH_CACHE: Dict[str, Any] = {}


def graph_of(db, k):
    if k not in _GRAPH_CACHE:
        _GRAPH_CACHE[k] = build_pure_graph_tensors(canonicalize_pd_snappy(db[k]["pd"]))
    return _GRAPH_CACHE[k]


def predict(model, graphs, chunk=256) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(graphs), chunk):
            outs.append(model(graphs[i:i + chunk]).numpy())
    return np.concatenate(outs, 0)


def train_regressor(backbone, db, tr, va, te, seed, epochs=None) -> Dict[str, Any]:
    epochs = epochs or EPOCHS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hidden = matched_hidden(backbone)
    Y = {s: np.stack([targets_of(db, ks, t)[0] for t in TARGETS], 1) for s, ks in (("tr", tr), ("va", va), ("te", te))}
    mean = np.nanmean(Y["tr"], 0)
    std = np.nanstd(Y["tr"], 0)
    std[std < 1e-6] = 1.0
    g = {s: [graph_of(db, k) for k in ks] for s, ks in (("tr", tr), ("va", va), ("te", te))}
    ytr = torch.tensor(np.nan_to_num((Y["tr"] - mean) / std), dtype=torch.float32)
    mtr = torch.tensor(~np.isnan(Y["tr"]), dtype=torch.float32)

    model = RegressionModel(backbone, len(TARGETS), hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = [float("inf")] * len(TARGETS)
    best_state: List[Optional[dict]] = [None] * len(TARGETS)
    idx = list(range(len(tr)))
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        random.shuffle(idx)
        for b in range(0, len(idx), BATCH):
            bi = idx[b:b + BATCH]
            pred = model([g["tr"][i] for i in bi])
            loss = (F.smooth_l1_loss(pred, ytr[bi], reduction="none") * mtr[bi]).sum() / mtr[bi].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        pv = predict(model, g["va"]) * std + mean
        for ti in range(len(TARGETS)):
            ok = ~np.isnan(Y["va"][:, ti])
            v = mae(pv[ok, ti], Y["va"][ok, ti])
            if v < best[ti]:
                best[ti] = v
                best_state[ti] = copy.deepcopy(model.state_dict())
    test_pred = np.zeros((len(te), len(TARGETS)))
    for ti in range(len(TARGETS)):
        model.load_state_dict(best_state[ti])
        test_pred[:, ti] = (predict(model, g["te"]) * std + mean)[:, ti]
    return {
        "backbone": backbone, "hidden": hidden, "params": count_params(model), "seed": seed,
        "train_sec": time.time() - t0, "val_mae": dict(zip(TARGETS, best)), "test_pred": test_pred,
    }


def tabular_models(Xtr, ytr, Xte, seed) -> Dict[str, np.ndarray]:
    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1).fit(Xtr, ytr)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=seed, subsample=0.8).fit(Xtr, ytr)
    return {"rf": rf.predict(Xte), "gbdt": gb.predict(Xte)}


def eval_split(db, feats, tr, va, te, backbones, seeds, tag) -> Dict[str, Any]:
    """Every baseline and model on one (train, val, test) split. Saves per-knot predictions."""
    res: Dict[str, Any] = {"split": {"train": len(tr), "val": len(va), "test": len(te)}, "targets": {}, "models": {}}
    preds: Dict[str, Dict[str, Any]] = {t: {} for t in TARGETS}
    Xd_tr, Xd_te = diagram_matrix(feats, tr), diagram_matrix(feats, te)
    for t in TARGETS:
        ytr, mtr = targets_of(db, tr, t)
        yte, mte = targets_of(db, te, t)
        P = preds[t]
        P["_y"] = yte.tolist()
        P["mean"] = [np.full(len(te), np.mean(ytr[mtr])).tolist()]
        P["linear_cw"] = [Ridge(alpha=1.0).fit(Xd_tr[mtr][:, [0, 3]], ytr[mtr]).predict(Xd_te[:, [0, 3]]).tolist()]
        Xo_tr, ocols = oracle_matrix(db, feats, tr, t)
        Xo_te, _ = oracle_matrix(db, feats, te, t)
        for s in seeds:
            for name, p in tabular_models(Xd_tr[mtr], ytr[mtr], Xd_te, s).items():
                P.setdefault(f"{name}_diagram", []).append(p.tolist())
            for name, p in tabular_models(Xo_tr[mtr], ytr[mtr], Xo_te, s).items():
                P.setdefault(f"{name}_oracle", []).append(p.tolist())
        # classical closed-form references (flagged: they read an invariant, not the diagram)
        sig = np.array([abs(float(db[k]["signature"] or 0)) / 2.0 for k in te])
        if t in ("unknotting_number", "smooth_four_genus"):
            P["bound_abs_sig_half"] = [sig.tolist()]
        if t == "three_genus":
            P["formula_alex_deg_half"] = [[(db[k]["alexander_vector"][1] - db[k]["alexander_vector"][0]) / 2.0 for k in te]]
        res["targets"][t] = {"n_test": int(mte.sum()), "oracle_columns": ocols}
    for bb in backbones:
        runs = []
        for s in seeds:
            r = train_regressor(bb, db, tr, va, te, s)
            print(f"  [{tag}] {bb:<8} seed {s} hidden={r['hidden']} params={r['params']} "
                  f"({r['train_sec']:.0f}s) " + " ".join(f"{t[:6]}={mae(r['test_pred'][~np.isnan(np.array(preds[t]['_y'])), i], np.array(preds[t]['_y'])[~np.isnan(np.array(preds[t]['_y']))]):.3f}" for i, t in enumerate(TARGETS)))
            runs.append(r)
            for i, t in enumerate(TARGETS):
                preds[t].setdefault(f"gnn_{bb}", []).append(r["test_pred"][:, i].tolist())
        res["models"][f"gnn_{bb}"] = {"hidden": runs[0]["hidden"], "params": runs[0]["params"],
                                      "train_sec_mean": float(np.mean([r["train_sec"] for r in runs]))}
    # summaries
    summary = {}
    for t in TARGETS:
        y = np.array(preds[t]["_y"], dtype=float)
        ok = ~np.isnan(y)
        alt = np.array([is_knot_alternating(k, db) for k in te])
        summary[t] = {}
        for name, plist in preds[t].items():
            if name.startswith("_"):
                continue
            arr = np.array(plist, dtype=float)
            per_seed = [mae(p[ok], y[ok]) for p in arr]
            summary[t][name] = {
                "mae_mean": float(np.mean(per_seed)), "mae_std": float(np.std(per_seed)), "n_seeds": len(per_seed),
                "mae_alternating": float(np.mean([mae(p[ok & alt], y[ok & alt]) for p in arr])),
                "mae_non_alternating": float(np.mean([mae(p[ok & ~alt], y[ok & ~alt]) for p in arr])),
            }
    res["summary"] = summary
    res["status"] = {
        "rf_oracle/gbdt_oracle": "ORACLE FEATURES: diagram features + sigma, |sigma|/2, det, Alexander. Sigma columns removed when the target is sigma.",
        "three_genus oracle": "FORMULA INPUT: Alexander degree/2 equals g3 for alternating knots.",
        "bound_abs_sig_half": "Classical lower bound read from KnotInfo sigma; reference, not a competitor.",
        "formula_alex_deg_half": "Classical formula; reference, not a competitor.",
    }
    res["_predictions"] = {"test_knots": te, "by_target": preds}
    return res


# =============================================================================
# Parts: task2, curve, davies
# =============================================================================

def part_task2(db, feats, backbones, seeds):
    tr, va, te = split_by_crossings(db, 0, 10), split_by_crossings(db, 11, 11), split_by_crossings(db, 12, 12)
    print(f"Task 2 headline: train {len(tr)} (c<=10) | val {len(va)} (c=11) | test {len(te)} (c=12)")
    res = eval_split(db, feats, tr, va, te, backbones, seeds, "task2")
    res["protocol"] = "train c<=10, per-target checkpoint selection on c=11, test c=12; no test-set tuning"
    save("task2_headline", res)


def part_curve(db, feats, backbones, seeds):
    c11 = sorted(split_by_crossings(db, 11, 11))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(c11))
    n_val = int(round(0.3 * len(c11)))
    va = [c11[i] for i in perm[:n_val]]
    c11_tr = [c11[i] for i in perm[n_val:]]
    te = split_by_crossings(db, 12, 12)
    sizes = {
        "c<=9": split_by_crossings(db, 0, 9),
        "c<=10": split_by_crossings(db, 0, 10),
        "c<=10+70%c11": split_by_crossings(db, 0, 10) + c11_tr,
    }
    out = {"val": f"fixed 30% of c=11 ({len(va)} knots, seed 0)", "test": f"c=12 ({len(te)})", "points": {}}
    for name, tr in sizes.items():
        print(f"Learning curve point {name}: {len(tr)} train knots")
        r = eval_split(db, feats, tr, va, te, backbones, seeds, f"curve {name}")
        out["points"][name] = {"n_train": len(tr), "summary": r["summary"]}
    save("learning_curve", out)


def round_to_even(x):
    return 2.0 * np.round(np.asarray(x) / 2.0)


def part_davies(db, feats, seeds):
    """Signature from hyperbolic invariants (Davies et al., Nature 2021)."""
    hyp_cols = ["volume", "chern_simons", "mu_re", "mu_im", "lam_re", "lam_im", "shape_re", "shape_im", "cusp_volume", "slope"]
    hyper = [k for k in db if feats[k]["hyperbolic"] is not None and db[k]["signature"] is not None]
    excluded = [k for k in db if feats[k]["hyperbolic"] is None]

    def X(ks):
        return np.array([[feats[k]["hyperbolic"][c] for c in hyp_cols] for k in ks])

    def y(ks):
        return np.array([float(db[k]["signature"]) for k in ks])

    def run(tr, te, label):
        Xtr, Xte, ytr, yte = X(tr), X(te), y(tr), y(te)
        out = {"n_train": len(tr), "n_test": len(te)}
        # Natural-slope formula: sigma ~ +/- slope/2 ; sign fit on training data only
        sgn = 1.0 if mae(Xtr[:, -1] / 2, ytr) < mae(-Xtr[:, -1] / 2, ytr) else -1.0
        p = sgn * Xte[:, -1] / 2.0
        out["natural_slope_formula"] = {"sign": sgn, "mae": mae(p, yte), "exact_acc": float(np.mean(round_to_even(p) == yte))}
        sc = StandardScaler().fit(Xtr)
        models = {
            "gbdt": lambda s: GradientBoostingRegressor(n_estimators=300, random_state=s, subsample=0.8),
            "rf": lambda s: RandomForestRegressor(n_estimators=300, random_state=s, n_jobs=1),
            "mlp": lambda s: MLPRegressor(hidden_layer_sizes=(300, 300, 300), max_iter=600, random_state=s, early_stopping=True),
        }
        for name, mk in models.items():
            maes, accs = [], []
            for s in seeds:
                m = mk(s).fit(sc.transform(Xtr), ytr)
                p = m.predict(sc.transform(Xte))
                maes.append(mae(p, yte))
                accs.append(float(np.mean(round_to_even(p) == yte)))
            out[name] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
                         "exact_acc_mean": float(np.mean(accs)), "exact_acc_std": float(np.std(accs))}
        out["mean_baseline_mae"] = mae(np.full(len(te), ytr.mean()), yte)
        print(f"  [davies {label}] " + json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in out.items()}))
        return out

    res = {"features": hyp_cols, "n_hyperbolic": len(hyper), "excluded_non_hyperbolic": excluded}
    tr = [k for k in hyper if db[k]["crossings"] <= 10]
    te = [k for k in hyper if db[k]["crossings"] == 12]
    res["crossing_split_c<=10_to_c=12"] = run(tr, te, "crossing split")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(hyper))
    cut = int(0.8 * len(hyper))
    res["random_80_20_split"] = run([hyper[i] for i in perm[:cut]], [hyper[i] for i in perm[cut:]], "random split")
    res["note"] = ("Davies et al. trained on a random split of a large census; the random_80_20 row mirrors that "
                   "setting within c<=12. The crossing split is the leakage-free KnotBench protocol. "
                   "Signature is never an input; features come only from the hyperbolic structure of the complement.")
    save("davies_reproduction", res)


# =============================================================================
# Part: task1 (hard-negative distinct presentations)
# =============================================================================

def presentations(db, k, rng_seed, steps=(10, 25, 40)) -> List[List[Tuple[int, int, int, int]]]:
    """Canonical (R1/R2-reduced, relabelled) diagrams of knot k from two KnotInfo presentations + backtracks."""
    out = []
    bases = [[tuple(c[:4]) for c in db[k]["pd"]]]
    b = parse_braid_word(db[k].get("braid"))
    if b:
        try:
            bases.append([tuple(c[:4]) for c in snappy.Link(braid_closure=b).PD_code()])
        except Exception:
            pass
    for bi, base in enumerate(bases):
        out.append(canonicalize_pd_snappy(safe_randomize_pd(base, seed=rng_seed + bi)[0]))
        for si, st in enumerate(steps):
            try:
                L = snappy.Link(list(base))
                random.seed(rng_seed + 31 * bi + si)
                L.backtrack(steps=st, prob_type_1=0.5, prob_type_2=0.5)
                pd = [tuple(c[:4]) for c in L.PD_code()]
            except Exception:
                pd = base
            out.append(canonicalize_pd_snappy(safe_randomize_pd(pd, seed=rng_seed + 1000 + 31 * bi + si)[0]))
    return [d for d in out if len(d) > 0]


def train_embedding(backbone, train_graphs: Dict[str, List[Any]], mode: str, seed: int, epochs=30) -> EmbeddingModel:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hidden = matched_hidden(backbone, kind="emb")
    names = sorted(train_graphs)
    model = EmbeddingModel(backbone, hidden=hidden, num_classes=len(names) if mode == "supervised" else 0)
    if mode == "untrained":
        return model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    if mode == "supervised":
        items = [(g, ci) for ci, n in enumerate(names) for g in train_graphs[n]]
        for _ in range(epochs):
            model.train()
            random.shuffle(items)
            for b in range(0, len(items), BATCH):
                batch = items[b:b + BATCH]
                _, logits = model([g for g, _ in batch])
                loss = F.cross_entropy(logits / 0.1, torch.tensor([c for _, c in batch]))
                opt.zero_grad()
                loss.backward()
                opt.step()
    else:  # contrastive NT-Xent over two distinct presentations of each knot
        multi = [n for n in names if len(train_graphs[n]) >= 2]
        for _ in range(epochs):
            model.train()
            random.shuffle(multi)
            for b in range(0, len(multi), BATCH):
                ks = multi[b:b + BATCH]
                pairs = [random.sample(train_graphs[k], 2) for k in ks]
                z1 = model.embed([p[0] for p in pairs])
                z2 = model.embed([p[1] for p in pairs])
                z = torch.cat([z1, z2], 0)
                sim = z @ z.t() / 0.1
                sim.fill_diagonal_(-1e9)
                n = len(ks)
                tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
                loss = F.cross_entropy(sim, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
    return model.eval()


def isometry_or_fallback(pd_a, braid_b) -> Tuple[float, str, float]:
    """1/0 certificate via exterior isometry; HFK equality fallback when not hyperbolic. Returns (score, method, seconds)."""
    t0 = time.perf_counter()
    try:
        A = snappy.Link(list(pd_a)).exterior()
        B = snappy.Link(braid_closure=braid_b).exterior()
        return (1.0 if A.is_isometric_to(B) else 0.0), "isometry", time.perf_counter() - t0
    except Exception:
        try:
            ha = snappy.Link(list(pd_a)).knot_floer_homology()
            hb = snappy.Link(braid_closure=braid_b).knot_floer_homology()
            same = ha["ranks"] == hb["ranks"] and ha["tau"] == hb["tau"]
            return (1.0 if same else 0.0), "hfk_fallback", time.perf_counter() - t0
        except Exception:
            return 0.5, "undetermined", time.perf_counter() - t0


def cluster_bootstrap_auroc(y, s, groups, reps=1000, seed=0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(reps):
        pick = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq))])
        if len(set(y[pick])) > 1:
            vals.append(roc_auc_score(y[pick], s[pick]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def part_task1(db, backbones, seeds, seed=42):
    # ---- test knots: identical selection to run_rigorous_tasks.py
    all_na = [k for k, v in db.items() if not is_knot_alternating(k, db) and v["crossings"] in (9, 10, 11, 12)]
    c910 = [k for k in all_na if db[k]["crossings"] in (9, 10)]
    c11 = [k for k in all_na if db[k]["crossings"] == 11]
    c12 = [k for k in all_na if db[k]["crossings"] == 12]
    c12s = list(np.random.default_rng(seed + 99).choice(c12, size=min(100, len(c12)), replace=False))
    test = []
    ki, br, braids = {}, {}, {}
    for k in c910 + c11 + c12s:
        b = parse_braid_word(db[k].get("braid"))
        if not b:
            continue
        try:
            pd_br = [tuple(c[:4]) for c in snappy.Link(braid_closure=b).PD_code()]
        except Exception:
            continue
        ki[k] = canonicalize_pd_snappy(safe_randomize_pd([tuple(c[:4]) for c in db[k]["pd"]], seed=seed + 10)[0])
        br[k] = canonicalize_pd_snappy(safe_randomize_pd(pd_br, seed=seed + 20)[0])
        braids[k] = b
        test.append(k)
    slices = {"c<=10": [k for k in test if db[k]["crossings"] <= 10],
              "c=11": [k for k in test if db[k]["crossings"] == 11],
              "c=12": [k for k in test if db[k]["crossings"] == 12]}

    # ---- training knots: knot-disjoint from the (non-alternating) test knots
    train_knots = [k for k, v in db.items() if 3 <= v["crossings"] <= 8] + \
                  [k for k, v in db.items() if v["crossings"] in (9, 10) and is_knot_alternating(k, db)]
    assert not set(train_knots) & set(test)
    print(f"Task 1: {len(train_knots)} training knots (c<=8 all + alternating c=9,10); test {[(s, len(v)) for s, v in slices.items()]}")
    train_graphs = {}
    for i, k in enumerate(train_knots):
        train_graphs[k] = [build_pure_graph_tensors(d) for d in presentations(db, k, 7000 + 13 * i)]

    # ---- fixed features and pairs
    geo_ki = {k: extract_diagram_geometric_summary(ki[k]) for k in test}
    geo_br = {k: extract_diagram_geometric_summary(br[k]) for k in test}
    t0 = time.perf_counter()
    wl_ki = {k: compute_wl_graph_hash(ki[k]) for k in test}
    wl_br = {k: compute_wl_graph_hash(br[k]) for k in test}
    wl_sec = (time.perf_counter() - t0) / (2 * len(test))

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

    rngp = np.random.default_rng(seed)
    pairs = {}  # (slice, mode) -> list of (anchor, other, label)
    for sname, ks in slices.items():
        for mode in ("random", "hard_geometric"):
            lst = []
            for k in ks:
                lst.append((k, k, 1))
                if mode == "random":
                    pool = [x for x in ks if x != k and db[x]["crossings"] == db[k]["crossings"]] or [x for x in ks if x != k]
                    negs = [rngp.choice(pool)]
                else:
                    sims = sorted(((cos(geo_ki[k], geo_br[o]), o) for o in ks if o != k), reverse=True)
                    negs = [o for _, o in sims[:2]]
                lst += [(k, n, 0) for n in negs]
            pairs[(sname, mode)] = lst

    # ---- certificate ceiling with runtime
    cert_cache, cert_meta = {}, []
    for lst in pairs.values():
        for a, b, _ in lst:
            if (a, b) not in cert_cache:
                sc, method, sec = isometry_or_fallback(db[a]["pd"], braids[b])
                cert_cache[(a, b)] = sc
                cert_meta.append({"pair": f"{a}|{b}", "method": method, "sec": sec})

    # ---- encoders
    enc_scores: Dict[str, List[Dict[str, float]]] = {}  # name -> per seed {pairkey: score}
    enc_info = {}
    for bb in backbones:
        for mode in ("untrained", "supervised", "contrastive"):
            name = f"{bb}_{mode}"
            enc_scores[name] = []
            for s in seeds:
                t0 = time.time()
                m = train_embedding(bb, train_graphs, mode, s)
                tr_sec = time.time() - t0
                with torch.no_grad():
                    t1 = time.perf_counter()
                    E_ki = dict(zip(test, m.embed([build_pure_graph_tensors(ki[k]) for k in test]).numpy()))
                    E_br = dict(zip(test, m.embed([build_pure_graph_tensors(br[k]) for k in test]).numpy()))
                    inf_sec = (time.perf_counter() - t1) / (2 * len(test))
                enc_scores[name].append({f"{a}|{b}": float(E_ki[a] @ E_br[b]) for lst in pairs.values() for a, b, _ in lst})
                enc_info.setdefault(name, {"params": count_params(m), "train_sec": [], "embed_sec_per_diagram": []})
                enc_info[name]["train_sec"].append(tr_sec)
                enc_info[name]["embed_sec_per_diagram"].append(inf_sec)
            print(f"  [task1] {name} done ({np.mean(enc_info[name]['train_sec']):.0f}s/seed)")

    # ---- evaluate
    table = {}
    pair_dump = {}
    for (sname, mode), lst in pairs.items():
        key = f"{sname}_{mode}"
        y = [l for _, _, l in lst]
        grp = [a for a, _, _ in lst]
        pk = [f"{a}|{b}" for a, b, _ in lst]
        row = {"num_knots": len(slices[sname]), "num_pairs": len(lst)}
        iso = [cert_cache[(a, b)] for a, b, _ in lst]
        row["certificate_ceiling"] = {"auroc": float(roc_auc_score(y, iso))}
        row["face_histogram"] = {"auroc": float(roc_auc_score(y, [cos(geo_ki[a], geo_br[b]) for a, b, _ in lst]))}
        row["wl_hash"] = {"auroc": float(roc_auc_score(y, [1.0 if wl_ki[a] == wl_br[b] else 0.0 for a, b, _ in lst]))}
        for name, per_seed in enc_scores.items():
            aucs = [float(roc_auc_score(y, [sd[p] for p in pk])) for sd in per_seed]
            mean_scores = np.mean([[sd[p] for p in pk] for sd in per_seed], 0)
            lo, hi = cluster_bootstrap_auroc(y, mean_scores, grp)
            row[name] = {"auroc_mean": float(np.mean(aucs)), "auroc_std": float(np.std(aucs)), "seeds": aucs,
                         "seed_mean_score_auroc_ci95_knot_bootstrap": [lo, hi]}
        table[key] = row
        pair_dump[key] = {"pairs": pk, "y": y, "groups": grp,
                          "scores": {n: np.mean([[sd[p] for p in pk] for sd in ps], 0).tolist() for n, ps in enc_scores.items()}}
        pair_dump[key]["scores"]["face_histogram"] = [cos(geo_ki[a], geo_br[b]) for a, b, _ in lst]
        print(f"  {key:<22} cert={row['certificate_ceiling']['auroc']:.3f} face={row['face_histogram']['auroc']:.3f} "
              + " ".join(f"{n}={row[n]['auroc_mean']:.3f}" for n in enc_scores))

    methods = [m["method"] for m in cert_meta]
    secs = np.array([m["sec"] for m in cert_meta])
    runtime = {
        "certificate_sec_per_pair": {"median": float(np.median(secs)), "mean": float(np.mean(secs)), "p95": float(np.percentile(secs, 95))},
        "certificate_methods": {m: methods.count(m) for m in set(methods)},
        "wl_hash_sec_per_diagram": wl_sec,
        "encoders": {n: {"params": v["params"], "train_sec_mean": float(np.mean(v["train_sec"])),
                         "embed_sec_per_diagram": float(np.mean(v["embed_sec_per_diagram"]))} for n, v in enc_info.items()},
    }
    save("task1_hard_negatives", {
        "train_set": "all knots 3<=c<=8 plus alternating knots c in {9,10}; knot-disjoint from the non-alternating test knots",
        "train_knots": len(train_knots), "slices": {s: len(v) for s, v in slices.items()},
        "encoders": "per backbone: untrained (frozen random), supervised (knot-ID cross-entropy), contrastive (NT-Xent over presentations); 5 seeds; parameter-matched",
        "certificate_note": "Exterior isometry (SnapPy) is a certificate of equivalence and is reported as the ceiling, not as a competitor. Non-hyperbolic pairs fall back to knot Floer homology equality (an invariant, not a certificate).",
        "results": table, "runtime": runtime,
    })
    save("task1_pair_scores", pair_dump)


# =============================================================================
# Part: gain (learned gain +0.008 in the c<=8 -> c=9 protocol)
# =============================================================================

def part_gain(seeds, epochs=20):
    import run_honest_protocol as hp
    from knot_recognition.knotinfo_bench import KnotPairSampler

    bench = hp.KnotInfoBenchmark()
    train_knots, test_9, test_10 = bench.get_splits()
    dataset = hp.prepare_benchmark_dataset(bench, num_train_diags=8, num_test_diags=4,
                                           cache_path=ROOT / "data" / "benchmark_diagrams_cache.json")
    knot_to_idx = {k: i for i, k in enumerate(train_knots)}
    device = torch.device("cpu")

    def build(key, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if key == "random_canon":
            m = hp.BaseKnotGraphModel(mode="r1_r2")
            m.eval()
            return m
        if key == "fixed_chord_canon":
            return hp.FixedChordFeatureBaseline(mode="r1_r2")
        if key == "shuffled_canon":
            return hp.train_graph_model(hp.BaseKnotGraphModel(mode="r1_r2"), dataset["train"], knot_to_idx,
                                        epochs=epochs, is_shuffled=True, device=device, seed=seed)[0]
        return hp.train_graph_model(hp.BaseKnotGraphModel(mode="r1_r2"), dataset["train"], knot_to_idx,
                                    epochs=epochs, device=device, seed=seed)[0]

    def pair_scores(emb, cands, split, seed):
        pairs = KnotPairSampler(bench).sample_pairs_for_dataset(split, cands, num_negatives_per_anchor=5, seed=seed)
        y, s, g = [], [], []
        for p in pairs:
            e1, e2 = emb.get(p.knot_1, []), emb.get(p.knot_2, [])
            if not e1 or not e2:
                continue
            a = e1[0]
            b = e2[1] if (p.is_positive and len(e2) > 1) else e2[0]
            y.append(1 if p.is_positive else 0)
            s.append(float(np.dot(a, b)))
            g.append(p.knot_1)
        return np.array(y), np.array(s), np.array(g)

    keys = ["canonical_r1r2", "random_canon", "shuffled_canon", "fixed_chord_canon"]
    per_seed = {k: [] for k in keys}
    for seed in seeds:
        for key in keys:
            m = build(key, seed)
            emb = hp.get_model_embeddings(m, dataset["test_unseen_9"], device=device)
            y, s, g = pair_scores(emb, test_9, dataset["test_unseen_9"], seed)
            per_seed[key].append({"y": y.tolist(), "s": s.tolist(), "g": g.tolist(), "auroc": float(roc_auc_score(y, s))})
            print(f"  [gain] seed {seed} {key}: AUROC {per_seed[key][-1]['auroc']:.4f}")
    save("learned_gain_pairs", {"protocol": "run_honest_protocol c<=8 train -> c=9 unseen, same sampler pairs per seed",
                                 "per_seed": per_seed})


# =============================================================================
# Part: stats
# =============================================================================

def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def paired_bootstrap_mae(y, pa, pb, reps=5000, seed=0):
    """Difference MAE(a) - MAE(b) with knot-level paired bootstrap."""
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(pa - y), np.abs(pb - y)
    d = ea - eb
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(reps)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], float(max(p, 1.0 / reps))


def paired_bootstrap_auroc(runs_a, runs_b, reps=2000, seed=0):
    """AUROC(a) - AUROC(b) averaged over seeds; anchors (knots) resampled jointly within each seed."""
    rng = np.random.default_rng(seed)
    obs = float(np.mean([ra["auroc"] - rb["auroc"] for ra, rb in zip(runs_a, runs_b)]))
    prepared = []
    for ra, rb in zip(runs_a, runs_b):
        assert ra["y"] == rb["y"] and ra["g"] == rb["g"], "runs must share pairs"
        y, g = np.array(ra["y"]), np.array(ra["g"])
        idx = {k: np.where(g == k)[0] for k in np.unique(g)}
        prepared.append((y, np.array(ra["s"]), np.array(rb["s"]), idx))
    boots = []
    for _ in range(reps):
        diffs = []
        for y, sa, sb, idx in prepared:
            keys = list(idx)
            pick = np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
            if len(set(y[pick])) < 2:
                continue
            diffs.append(roc_auc_score(y[pick], sa[pick]) - roc_auc_score(y[pick], sb[pick]))
        boots.append(np.mean(diffs))
    boots = np.array(boots)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    seed_diffs = [ra["auroc"] - rb["auroc"] for ra, rb in zip(runs_a, runs_b)]
    return {"diff": obs, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_boot": float(max(p, 1.0 / reps)), "seed_diffs": seed_diffs,
            "seeds_positive": int(sum(d > 0 for d in seed_diffs))}


def part_stats():
    out: Dict[str, Any] = {"method": {
        "task2": "Paired knot-level bootstrap (5,000 resamples) on seed-averaged predictions; MAE(A) - MAE(B); two-sided p; Holm correction across the whole family.",
        "task1": "Knot-cluster paired bootstrap (2,000 resamples) of AUROC differences on seed-averaged scores; Holm-corrected.",
        "learned_gain": "Per seed, both models are scored on identical sampler pairs; anchors resampled jointly; differences averaged over 5 seeds.",
    }}
    raw_p: Dict[str, float] = {}
    t2p = OUT_DIR / "task2_headline.json"
    if t2p.exists():
        t2 = json.load(open(t2p))
        P = t2["_predictions"]["by_target"]
        comps = {}
        for t in TARGETS:
            y = np.array(P[t]["_y"], dtype=float)
            ok = ~np.isnan(y)
            avg = {n: np.mean(np.array(v, dtype=float), 0)[ok] for n, v in P[t].items() if not n.startswith("_")}
            gnns = [n for n in avg if n.startswith("gnn_")]
            best_gnn = min(gnns, key=lambda n: mae(avg[n], y[ok]))
            pairs = [(g, "rf_diagram") for g in gnns] + [(g, "linear_cw") for g in gnns] + \
                    [(best_gnn, "gnn_mpnn")] * (best_gnn != "gnn_mpnn") + [("rf_oracle", "rf_diagram"), ("rf_diagram", "linear_cw")]
            for a, b in pairs:
                d, ci, p = paired_bootstrap_mae(y[ok], avg[a], avg[b])
                key = f"task2/{t}/{a}_vs_{b}"
                comps[key] = {"mae_diff": d, "ci95": ci, "p": p}
                raw_p[key] = p
        out["task2"] = comps
    t1p = OUT_DIR / "task1_pair_scores.json"
    if t1p.exists():
        t1 = json.load(open(t1p))
        comps = {}
        for key, blob in t1.items():
            if not key.endswith("hard_geometric"):
                continue
            y, g = np.array(blob["y"]), np.array(blob["groups"])
            sc = {n: np.array(v) for n, v in blob["scores"].items()}
            trained = [n for n in sc if n.endswith("_supervised") or n.endswith("_contrastive")]
            best = max(trained, key=lambda n: roc_auc_score(y, sc[n]))
            bb = best.split("_")[0]
            for a, b in [(best, f"{bb}_untrained"), (best, "face_histogram")]:
                r = paired_bootstrap_auroc([{"y": y.tolist(), "g": g.tolist(), "s": sc[a].tolist(), "auroc": roc_auc_score(y, sc[a])}],
                                           [{"y": y.tolist(), "g": g.tolist(), "s": sc[b].tolist(), "auroc": roc_auc_score(y, sc[b])}])
                k2 = f"task1/{key}/{a}_vs_{b}"
                comps[k2] = r
                raw_p[k2] = r["p_boot"]
        out["task1"] = comps
    gp = OUT_DIR / "learned_gain_pairs.json"
    if gp.exists():
        G = json.load(open(gp))["per_seed"]
        comps = {}
        for a, b in [("canonical_r1r2", "random_canon"), ("canonical_r1r2", "shuffled_canon"), ("canonical_r1r2", "fixed_chord_canon")]:
            r = paired_bootstrap_auroc(G[a], G[b])
            r["auroc_a_mean"] = float(np.mean([x["auroc"] for x in G[a]]))
            r["auroc_b_mean"] = float(np.mean([x["auroc"] for x in G[b]]))
            k2 = f"gain/{a}_vs_{b}"
            comps[k2] = r
            raw_p[k2] = r["p_boot"]
        out["learned_gain"] = comps
    adj = holm(raw_p)
    for sec in ("task2", "task1", "learned_gain"):
        for k, v in out.get(sec, {}).items():
            v["p_holm"] = adj[k]
    out["family_size"] = len(raw_p)
    save("statistics", out)


# =============================================================================

def main():
    global EPOCHS, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", choices=["features", "task2", "curve", "davies", "task1", "gain", "stats", "all"])
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="graph-regression epochs (smoke tests only)")
    ap.add_argument("--tag", default="", help="suffix for output files (smoke tests only)")
    a = ap.parse_args()
    EPOCHS = a.epochs
    if a.tag:
        OUT_DIR = OUT_DIR.parent / f"benchmark_v2_{a.tag}"
    parts = ["features", "task2", "curve", "davies", "task1", "gain", "stats"] if "all" in a.parts else a.parts
    db = load_db()
    feats = part_features(db) if any(p in parts for p in ("features", "task2", "curve", "davies")) else None
    for p in parts:
        t0 = time.time()
        print(f"\n===== {p} =====")
        if p == "task2":
            part_task2(db, feats, a.backbones, a.seeds)
        elif p == "curve":
            part_curve(db, feats, a.backbones, a.seeds)
        elif p == "davies":
            part_davies(db, feats, a.seeds)
        elif p == "task1":
            part_task1(db, a.backbones, a.seeds)
        elif p == "gain":
            part_gain(a.seeds)
        elif p == "stats":
            part_stats()
        print(f"===== {p} done in {time.time()-t0:.0f}s =====")


if __name__ == "__main__":
    main()
