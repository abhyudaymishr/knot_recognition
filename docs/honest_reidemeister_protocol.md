# Certified Reidemeister Invariance vs. Topological Equivalence

## 1. Executive Summary & Honest Scientific Claim

> **The Honest Claim**:  
> *"A canonicalization layer gives exact $R_1 / R_2$ invariance; here is what that buys a downstream learned model."*

In knot theory and geometric topology, knot equivalence is defined modulo ambient isotopy, which Reidemeister's Theorem decomposes into planar isotopies and three elementary local moves:
1. **$R_1$ (Twist / Curl)**: Markov stabilization / destabilization.
2. **$R_2$ (Poke / Bigon)**: Cancelling pair introduction / deletion.
3. **$R_3$ (Slide / Triangle)**: Artin braid Yang-Baxter slide ($\sigma_i \sigma_{i+1} \sigma_i \longleftrightarrow \sigma_{i+1} \sigma_i \sigma_{i+1}$).

Crucially, **a confluent canonicalization layer can only eliminate $R_1$ curls and $R_2$ bigons in polynomial time $O(|V|)$**. It does **not** solve full knot equivalence. The hard part of knot equivalence lies in:
- **$R_3$ moves**: Triangle slides do not change the crossing count ($\Delta c = 0$) but alter planar strand routing and crossing neighborhoods. They cannot be eliminated by greedy local reductions without risking exponential state spaces or cycle loops.
- **Flypes**: In alternating and 2-bridge knots, 180° rotations of 2-tangles (Tait's Flyping Conjecture, proven by Menasco & Thistlethwaite in 1991) transform minimal diagrams into distinct isotopic diagrams without changing the crossing number.

Therefore, we do not claim that canonicalization "solves knot recognition." We rigorously test: **when an exact $R_1/R_2$ canonicalization layer reduces isotopic diagrams to their normal form, does the downstream neural encoder generalize better to unseen knots and to hard isotopic diagrams (perturbed by $R_3$ and flypes)?**

---

## 2. Experimental Design & Split Strategy

To ensure zero information leakage and prevent trivial visual memorization:

### 2.1 Unified Representation Across All Graph Models
- **Exact Same Input**: All graph models receive the signed 4-valent Planar Diagram (PD) code or signed planar graph $G = (V, E)$, with node crossing signs $s_v \in \{-1, +1\}$ and arc port incident attributes.
- **Rendered-Image ViT Reference Row**: Evaluated strictly as a separate reference baseline, since vision models process pixel grids rather than topological graphs.

### 2.2 Knot-Level Split (Not Diagram-Level Split)
- **Train Set**: Knots with crossing number $c(K) \le 8$ (36 prime knots + unknot = 37 knots). Mirror images are strictly retained in the same split.
- **Test Set 1 (Unseen Knots, $c = 9$)**: All 49 prime knots with $c = 9$.
- **Test Set 2 (Unseen Knots, $c = 10$)**: All 165 prime knots with $c = 10$.

### 2.3 Two Independent Evaluation Paradigms
- **Test Type A (Seen Knots, New Diagrams)**: Held-out diagrams of the 37 training knots ($c \le 8$) perturbed by heavy combinations of $R_1, R_2, R_3$, and flypes up to $\min(c) + 6$ crossings. Evaluates diagram invariance on known topologies.
- **Test Type B (Unseen Knots)**: Diagrams of completely unseen knots ($c = 9$ and $c = 10$). Evaluates inductive generalization to higher crossing numbers.

---

## 3. Evaluation Tasks & Hard Negatives

### 3.1 Task 1: Same-Knot Verification
Given diagram pairs $(D_1, D_2)$, verify whether $[K(D_1)] = [K(D_2)]$ using cosine similarity $s = \frac{z_1 \cdot z_2}{\|z_1\| \|z_2\|}$.
- **Positive Pairs**: Two distinct isotopic diagrams of the same knot perturbed by $R_3$ slides, flypes, and up to 6 extra $R_1/R_2$ crossings.
- **Hard Negatives Pool**: Mined strictly from knots sharing:
  1. **Identical Alexander polynomial $\Delta(t)$** (e.g., $8_9$ vs $4_1 \# 4_1$, or pairs in census sharing $\Delta(t)$).
  2. **Identical Jones polynomial $V(t)$**.
  3. **Identical crossing number $c(K)$** (e.g. distinct 9-crossing knots).
- **Metrics**: AUROC, Top-1 Retrieval Accuracy, and Top-5 Retrieval Accuracy.

### 3.2 Task 2: Invariant Prediction on Unseen Knots
Evaluates linear probes trained on $c \le 8$ representations to predict exact topological invariants from **KnotInfo** on unseen $c = 9$ and $c = 10$ knots:
1. **Signature $\sigma(K)$**: Integer $\in 2\mathbb{Z}$ (no cheap formula from raw diagram without Seifert surface or matrix diagonalization).
2. **Three-Genus $g_3(K)$**: Minimal genus of a Seifert surface bounded by $K$.
3. **Braid Index $b(K)$**: Minimal number of strands needed to close to $K$.
4. **Unknotting Number $u(K)$**: Famously NP-hard to compute; unknown even for several 10-crossing knots.

### 3.3 Trivial Baseline Predictors
Every learned model is benchmarked against:
- **Mean Baseline**: Predicts the training set target mean for all test knots.
- **Linear Fit on $[c(K), w(K)]$**: Ridge regression on diagram crossing number $c(K)$ and diagram writhe $w(K)$.

---

## 4. Models and Ablation Suite

All models share the **exact same 4-layer message-passing GNN backbone** ($d = 128$) with LayerNorm, GELU, and residual connections, trained with the identical budget across 5 random seeds ($[42, 43, 44, 45, 46]$):

| Model Category | Model Name | Description |
| :--- | :--- | :--- |
| **Core Models** | `Raw GNN` | Same GNN without canonicalization layer. |
| | `Canonical GNN (R1/R2)` | GNN preceded by confluent $R_1$ curl and $R_2$ bigon reduction. |
| | `Augmented GNN` | Raw GNN trained with stochastic online $R_1, R_2, R_3$, flype augmentation. |
| | `Contrastive GNN` | GNN trained with multi-negative InfoNCE loss over isotopic pairs. |
| **Reference Row** | `Rendered ViT` | NanoTransformer (patch 8x8) trained on 64x64 rendered knot diagrams. |
| **Baselines** | `Non-Learned Baseline` | Exact Alexander polynomial evaluations plus crossing count and writhe. |
| | `Lookup-Table Hash` | Hashes canonical R-NF PD code; scores near zero on unseen knots. |
| **Ablations** | `Canonical GNN (R1-only)` | Ablation: removes $R_2$ bigon collapse, keeping only $R_1$ curls. |
| | `Canonical GNN (R2-only)` | Ablation: removes $R_1$ curl contraction, keeping only $R_2$ bigons. |
| | `SnapPy Simplify` | Upper bound: utilizes SnapPy's `Link.simplify('level')` with $R_3$-guided simplification. |
| **Sanity Controls** | `Shuffled-Label Control` | GNN trained with randomly permuted knot identities. |
| | `Random-Feature Control` | Untrained GNN with frozen randomly initialized weights. |

---

## 5. Decision Rule: Support vs. Refutation

Following open, honest scientific inquiry:
- **Support**: Canonicalization closes most of the gap on same-knot verification under $R_3$/flypes **AND** the downstream learned model outperforms trivial predictors ($c(K), w(K)$ linear fit) on at least one topological invariant on unseen crossing numbers ($c=9, 10$).
- **Refutation / Negative Finding**: All encoders converge to similar scores, or the model fails to outperform trivial crossing-number predictors on unseen knots. In this case, the honest paper presents a negative and analytical finding demonstrating the exact mathematical limitations of graph message passing under $R_3$ slides.
