"""
make_paper_tables.py - Write every LaTeX table in the paper from results/benchmark_v2/*.json.

No number in paper/neurips/tables/ is typed by hand. Run after run_benchmark_v2.py:
    python scripts/make_paper_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "benchmark_v2"
OUT = ROOT / "paper" / "neurips" / "tables"

TNAME = {"unknotting_number": "$u(K)$", "smooth_four_genus": "$g_4(K)$", "three_genus": "$g_3(K)$", "signature": "$\\sigma(K)$"}
BB = ["mpnn", "gin", "gps", "subgraph"]
BBNAME = {"mpnn": "Signed MPNN", "gin": "GIN", "gps": "GPS", "subgraph": "Subgraph (DS-GNN)"}


def load(name):
    p = RES / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def f3(x):
    return f"{x:.3f}"


def pm(m, s):
    return f"${m:.3f}\\pm{s:.3f}$"


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)
    print(f"[table] {OUT / (name + '.tex')}")


def table_task2():
    r = load("task2_headline")
    st = load("statistics") or {}
    if not r:
        return
    S = r["summary"]
    cols = ["mean", "linear_cw", "rf_diagram", "gbdt_diagram"] + [f"gnn_{b}" for b in BB]
    head = ["Mean", "Lin.\\ $[c,w]$", "RF", "GBDT"] + [BBNAME[b] for b in BB]
    lines = [
        "\\begin{tabular}{l r " + "c" * len(cols) + "}", "\\toprule",
        " & & \\multicolumn{2}{c}{Trivial} & \\multicolumn{2}{c}{Diagram-only trees} & \\multicolumn{4}{c}{Graph backbones (param.-matched)}\\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-10}",
        "Target & $N$ & " + " & ".join(head) + "\\\\", "\\midrule",
    ]
    for t in ["unknotting_number", "smooth_four_genus", "three_genus", "signature"]:
        vals = {c: S[t][c] for c in cols}
        best = min(vals, key=lambda c: vals[c]["mae_mean"])
        cells = []
        for c in cols:
            v = vals[c]
            txt = f3(v["mae_mean"]) if v["n_seeds"] == 1 or v["mae_std"] < 1e-9 else pm(v["mae_mean"], v["mae_std"])
            cells.append(f"\\textbf{{{txt}}}" if c == best else txt)
        lines.append(f"{TNAME[t]} & {r['targets'][t]['n_test']} & " + " & ".join(cells) + "\\\\")
    lines += ["\\midrule", "\\multicolumn{10}{l}{\\emph{Oracle inputs (not competitors): trees given $\\sigma$, $\\det$, Alexander coefficients; classical formulas}}\\\\"]
    for t in ["unknotting_number", "smooth_four_genus", "three_genus", "signature"]:
        o = S[t]
        ref = o.get("bound_abs_sig_half") or o.get("formula_alex_deg_half")
        ref_txt = f"; closed form {f3(ref['mae_mean'])}" if ref else ""
        flag = " (formula input)" if t == "three_genus" else (" ($\\sigma$ columns removed)" if t == "signature" else "")
        lines.append(f"\\flag{{{TNAME[t]}}} & & \\multicolumn{{8}}{{l}}{{\\flag{{RF {pm(o['rf_oracle']['mae_mean'], o['rf_oracle']['mae_std'])}, "
                     f"GBDT {pm(o['gbdt_oracle']['mae_mean'], o['gbdt_oracle']['mae_std'])}{flag}{ref_txt}}}}}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("task2", "\n".join(lines) + "\n")


def table_task1():
    r = load("task1_hard_negatives")
    if not r:
        return
    R = r["results"]
    rows = [("c<=10", "$c\\le10$"), ("c=11", "$c=11$"), ("c=12", "$c=12$")]
    lines = ["\\begin{tabular}{l l c c c " + "c" * len(BB) + "}", "\\toprule",
             "Slice & Neg. & Ceiling & Face hist. & Untrained & " + " & ".join(BBNAME[b] for b in BB) + "\\\\",
             "\\midrule"]
    for key, lab in rows:
        for mode, mlab in (("random", "random"), ("hard_geometric", "hard")):
            row = R[f"{key}_{mode}"]
            untr = max((row[f"{b}_untrained"] for b in BB), key=lambda v: v["auroc_mean"])
            cells = []
            for b in BB:
                best = max((row[f"{b}_supervised"], row[f"{b}_contrastive"]), key=lambda v: v["auroc_mean"])
                cells.append(pm(best["auroc_mean"], best["auroc_std"]))
            lines.append(f"{lab} ({row['num_knots']}) & {mlab} & {f3(row['certificate_ceiling']['auroc'])} & "
                         f"{f3(row['face_histogram']['auroc'])} & {pm(untr['auroc_mean'], untr['auroc_std'])} & " + " & ".join(cells) + "\\\\")
    rt = r["runtime"]
    lines += ["\\bottomrule", "\\end{tabular}"]
    note = (f"% certificate: median {rt['certificate_sec_per_pair']['median']*1e3:.1f} ms/pair; "
            f"methods {rt['certificate_methods']}\n")
    write("task1", note + "\n".join(lines) + "\n")


def table_runtime():
    r = load("task1_hard_negatives")
    if not r:
        return
    rt = r["runtime"]
    lines = ["\\begin{tabular}{l r r}", "\\toprule", "Method & Params & Time per diagram / pair\\\\", "\\midrule",
             f"Exterior isometry (certificate) & -- & {rt['certificate_sec_per_pair']['median']*1e3:.1f} ms (median), {rt['certificate_sec_per_pair']['p95']*1e3:.1f} ms (p95)\\\\",
             f"WL hash & -- & {rt['wl_hash_sec_per_diagram']*1e3:.2f} ms\\\\"]
    for b in BB:
        e = rt["encoders"].get(f"{b}_supervised")
        if e:
            lines.append(f"{BBNAME[b]} encoder & {e['params']:,} & {e['embed_sec_per_diagram']*1e3:.2f} ms\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("runtime", "\n".join(lines) + "\n")


def table_curve():
    r = load("learning_curve")
    if not r:
        return
    pts = r["points"]
    names = list(pts)
    lines = ["\\begin{tabular}{l l " + "c" * len(names) + "}", "\\toprule",
             "Target & Model & " + " & ".join(f"{n.replace('<=', '$\\le$').replace('%', '\\%')} ({pts[n]['n_train']})" for n in names) + "\\\\", "\\midrule"]
    for t in ["unknotting_number", "smooth_four_genus", "three_genus", "signature"]:
        for m in ["linear_cw", "rf_diagram"] + [f"gnn_{b}" for b in BB]:
            lab = {"linear_cw": "Lin.\\ $[c,w]$", "rf_diagram": "RF (diagram)"}.get(m, BBNAME.get(m[4:], m))
            cells = [pm(pts[n]["summary"][t][m]["mae_mean"], pts[n]["summary"][t][m]["mae_std"]) for n in names]
            lines.append(f"{TNAME[t] if m == 'linear_cw' else ''} & {lab} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule" if t != "signature" else "")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("learning_curve", "\n".join(lines) + "\n")


def table_davies():
    r = load("davies_reproduction")
    if not r:
        return
    lines = ["\\begin{tabular}{l c c c c}", "\\toprule", " & \\multicolumn{2}{c}{Random 80/20 (Davies et al.\\ setting)} & \\multicolumn{2}{c}{KnotBench $c\\le10\\to c=12$}\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Predictor of $\\sigma(K)$ & MAE & Exact acc. & MAE & Exact acc.\\\\", "\\midrule"]
    A, B = r["random_80_20_split"], r["crossing_split_c<=10_to_c=12"]
    lines.append(f"Mean & {f3(A['mean_baseline_mae'])} & -- & {f3(B['mean_baseline_mae'])} & --\\\\")
    lines.append(f"Natural slope / 2 (formula) & {f3(A['natural_slope_formula']['mae'])} & {A['natural_slope_formula']['exact_acc']:.3f} & "
                 f"{f3(B['natural_slope_formula']['mae'])} & {B['natural_slope_formula']['exact_acc']:.3f}\\\\")
    for m, lab in (("mlp", "MLP (hyperbolic inv.)"), ("gbdt", "GBDT (hyperbolic inv.)"), ("rf", "RF (hyperbolic inv.)")):
        lines.append(f"{lab} & {pm(A[m]['mae_mean'], A[m]['mae_std'])} & {A[m]['exact_acc_mean']:.3f} & "
                     f"{pm(B[m]['mae_mean'], B[m]['mae_std'])} & {B[m]['exact_acc_mean']:.3f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("davies", "\n".join(lines) + "\n")


def table_stats():
    s = load("statistics")
    if not s:
        return
    lines = ["\\begin{longtable}{p{8.2cm} r c r r}", "\\toprule", "Comparison (A vs B) & $\\Delta$ & 95\\% CI & $p$ & $p_\\text{Holm}$\\\\", "\\midrule", "\\endhead"]
    for sec in ("learned_gain", "task1", "task2"):
        for k, v in s.get(sec, {}).items():
            d = v.get("diff", v.get("mae_diff"))
            lab = k.replace("_vs_", " vs ").replace("/", " / ").replace("_", "\\_").replace("<=", "$\\le$")
            lines.append(f"{lab} & {d:+.3f} & [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] & "
                         f"{v.get('p', v.get('p_boot')):.4f} & {v['p_holm']:.3f}\\\\")
    lines += ["\\bottomrule", "\\end{longtable}"]
    write("statistics", "\n".join(lines) + "\n")


if __name__ == "__main__":
    for fn in (table_task2, table_task1, table_runtime, table_curve, table_davies, table_stats):
        fn()
