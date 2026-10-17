"""
eval_recheck.py - The head's links with and without re-checking the jumps (protocol 13.6, 13.7).

Variants: head alone; the straight check (check_jumps, balanced 70 deg / ridge mean 0.3, and strict
40 deg / 0.4 / min 0.1); the curved re-check with re-linking (verify_relink); and the curved re-check
followed by each straight setting (recheck = the balanced one). Per point source: jumps found, wrong
long links (links longer than two spacings that are not true), images with no wrong long link,
images with every link right.

  python scripts/eval_recheck.py 0 150                       # development images
  python scripts/eval_recheck.py 150 150 results/strand_pointer/recheck_combined_heldout.json
  python scripts/eval_recheck.py 150 150 out.json --head checkpoints/uf_pointer/mps_ufptr_s0_head.pth
"""
import sys, json, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]
warnings.filterwarnings("ignore")
import numpy as np, torch
from knot_recognition.strand_pointer import *
from knot_recognition.knot_aware import load
from train_strand_pointer import images, make_points
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("start", type=int); ap.add_argument("n", type=int); ap.add_argument("out", nargs="?", default=None)
ap.add_argument("--head", default=str(ROOT / "checkpoints/strand_pointer/smoke3_s0.pth"))
args = ap.parse_args()
start, n = args.start, args.n
head=load_head(args.head); trunk=load(ROOT/head.extra["trunk"]).eval()
recs=load_records(ROOT/"data/knot_aware/val_v3_strands.pkl"); X=images("val_v3"); sp=head.cfg.spacing
B=dict(max_angle=70.0, ridge_mean=0.3, ridge_min=0.0); S_=dict(max_angle=40.0, ridge_mean=0.4, ridge_min=0.1)
names=["head alone","straight balanced","straight strict","curved relink","curved relink + straight balanced","curved relink + straight strict"]
agg={m:{k:[0,0,0,0,0] for k in names} for m in ("true","ridge")}
for i in range(start,start+n):
    with torch.no_grad():
        out=trunk(torch.from_numpy(np.array(X[i:i+1])).permute(0,3,1,2).float()/255,return_feats=True)
    ridge=torch.sigmoid(out["center_hi"][0]).numpy()
    for mode in ("true","ridge"):
        pts,C,w=make_points(out,0,recs[i],mode,None,0.0,10**6,sp); R=cut_radius(w)
        xy,valid,_,_=targets([pts])
        with torch.no_grad(): p=torch.softmax(head(out["feats"],xy,valid,torch.tensor([R])),-1)[0].numpy()
        E0=decode(p); E1=verify_relink(E0,p,pts.xy,ridge,R,sp)
        cj=lambda E,kw: [tuple(e) for e in check_jumps(E,pts.xy,ridge,sp,**kw)[0]]
        res={"head alone":E0,"straight balanced":cj(E0,B),"straight strict":cj(E0,S_),"curved relink":E1,
             "curved relink + straight balanced":cj(E1,B),"curved relink + straight strict":cj(E1,S_)}
        T={frozenset(e[:2]) for e in pts.edges}; J={frozenset(e[:2]) for e in pts.edges if e[2]}
        for name,E in res.items():
            S={frozenset(e[:2]) for e in E}
            long=[e for e in E if np.linalg.norm(pts.xy[e[0]]-pts.xy[e[1]])>2*sp]
            wrong=sum(frozenset(e[:2]) not in T for e in long)
            a=agg[mode][name]; a[0]+=len(J&S); a[1]+=len(J); a[2]+=wrong; a[3]+=wrong==0; a[4]+=int(S==T)
out={}
for m,d in agg.items():
    for k,a in d.items():
        print(f"{m:6s} {k:36s} jumps {a[0]/a[1]:.3f}  wrong long {a[2]:4d}  images none wrong {a[3]:3d}/{n}  every link right {a[4]:3d}")
        out.setdefault(m,{})[k]={"jumps_found":a[0]/a[1],"wrong_long_links":a[2],"images_without_wrong_long_link":a[3],"images_every_link_right":a[4]}
if args.out: json.dump({"head":args.head,"images":[start,start+n],"results":out},open(args.out,"w"),indent=1)
