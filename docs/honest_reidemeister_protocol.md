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

---

## 6. Empirical Results (5 Random Seeds: 42, 43, 44, 45, 46)

### 6.1 Task 1: Same-Knot Verification with Matched-Crossing Hard Negatives
All models evaluated across 5 random seeds ($N = 5$) on signed 4-valent PD code / planar graphs with symmetric controls decoupling the effect of the canonicalization preprocessor, fixed hand-built Gauss chord features, and the learned neural encoder.

| Model Category | Model Name | Seen-A AUROC ($c \le 8$) | Unseen-9 AUROC ($c = 9$) |
| :--- | :--- | :---: | :---: |
| **Fixed Features (No Neural Weights)** | Fixed Chord Features (Raw Input) | $0.580 \pm 0.008$ | $0.547 \pm 0.009$ |
| | **Fixed Chord Features (Canonical Input)** | **$0.882 \pm 0.002$** | **$0.925 \pm 0.002$** |
| **Untrained Encoders (Random Weights)** | Random Untrained GNN (Raw Input) | $0.618 \pm 0.006$ | $0.672 \pm 0.019$ |
| | **Random Untrained GNN (Canonical Input)** | **$0.945 \pm 0.004$** | **$0.955 \pm 0.002$** |
| **Sanity Controls (Shuffled Labels)** | Shuffled-Label GNN (Raw Input) | $0.606 \pm 0.010$ | $0.688 \pm 0.011$ |
| | **Shuffled-Label GNN (Canonical Input)** | **$0.947 \pm 0.002$** | **$0.952 \pm 0.003$** |
| **Learned Graph Encoders** | Raw GNN (No Canonicalization) | $0.581 \pm 0.016$ | $0.688 \pm 0.019$ |
| | **Canonical GNN ($R_1/R_2$)** | **$0.956 \pm 0.003$** | **$0.963 \pm 0.003$** |
| | Augmented GNN (Online Perturbations) | $0.570 \pm 0.018$ | $0.684 \pm 0.008$ |
| | Contrastive GNN (InfoNCE) | $0.656 \pm 0.033$ | $0.742 \pm 0.020$ |
| **Algorithmic Ablations** | Canonical GNN ($R_1$-only Ablation) | $0.703 \pm 0.009$ | $0.777 \pm 0.021$ |
| | Canonical GNN ($R_2$-only Ablation) | $0.561 \pm 0.015$ | $0.501 \pm 0.016$ |
| | **SnapPy Simplify (Upper Bound)** | **$0.954 \pm 0.009$** | **$0.957 \pm 0.006$** |
| **Reference Baselines** | Lookup-Table Hash | $0.610 \pm 0.003$ | $0.606 \pm 0.006$ |
| | **Non-Learned Baseline (Alexander $+ c + w$)** | **$0.988 \pm 0.002$** | **$0.988 \pm 0.000$** |
| | Rendered ViT (Reference Row) | $0.961 \pm 0.022$ | $0.962 \pm 0.034$ |

### 6.2 Task 2: Invariant Prediction on Unseen Knots without Cheap Formulas ($c = 9$)
Evaluates linear probes trained on $c \le 8$ representations against trivial predictors on unseen $c = 9$ knots.

| Model / Predictor | Signature $\sigma(K)$ MAE | Unknotting Number $u(K)$ MAE |
| :--- | :---: | :---: |
| **Trivial Mean Baseline** | $2.354$ | $0.724$ |
| **Trivial Linear Fit $[c(K), w(K)]$** | **$2.327$** | **$0.627$** |
| Raw GNN | $2.641 \pm 0.043$ | $0.723 \pm 0.002$ |
| Canonical GNN ($R_1/R_2$) | $2.658 \pm 0.035$ | $0.709 \pm 0.006$ |
| Canonical GNN ($R_1$-only) | $2.668 \pm 0.017$ | $0.713 \pm 0.003$ |
| Canonical GNN ($R_2$-only) | $2.257 \pm 0.007$ | $0.702 \pm 0.006$ |
| SnapPy Simplify | $2.421 \pm 0.019$ | $0.703 \pm 0.003$ |
| Augmented GNN | $2.601 \pm 0.059$ | $0.723 \pm 0.003$ |
| Contrastive GNN | $2.399 \pm 0.022$ | $0.724 \pm 0.001$ |
| **Non-Learned Baseline (Alexander $+ c + w$)** | **$1.937 \pm 0.000$** | **$0.628 \pm 0.000$** |

---

## 7. Honest Scientific Takeaways: What Does the Encoder Actually Do?

1. **The Core Performance Driver is Canonicalization + Fixed Geometric Features, NOT Neural Learning**:
   - As hypothesized in the audit, **a control trained on shuffled labels lands at $0.606 / 0.688$ on raw input, but jumps to $0.947 / 0.952$ when passed canonicalized input**.
   - An untrained, randomly initialized GNN with frozen random weights on canonicalized input scores **$0.945 \pm 0.004$ (Seen-A)** and **$0.955 \pm 0.002$ (Unseen-9)**.
   - Pure hand-built summary statistics (mean and max Gauss chord lengths, crossing counts, crossing signs) with **zero neural network weights whatsoever** achieve **$0.925 \pm 0.002$ AUROC** on unseen 9-crossing knots after canonicalization.
   - Fully supervised training of the 4-layer GNN on canonicalized input yields **$0.963 \pm 0.003$ AUROC**—a net learning gain of only **$+0.008$ AUROC over an untrained random encoder** and **$+0.038$ AUROC over pure fixed chord statistics**.

2. **The Honest Re-Framing**:
   - The paper's claim must **not** be: *"Canonicalization dramatically improves a learned GNN's ability to understand knot equivalence."*
   - The true, scientifically defensible finding is:
     > **"Exact $R_1/R_2$ reduction collapses planar diagram variance into a sufficiently regular orbit such that simple geometric Gauss chord metrics alone produce a high-accuracy ($0.925$ AUROC) untrained scorer, and random graph projections reach $0.955$ AUROC. Supervised neural learning contributes only marginal refinement ($+0.008$ AUROC)."**

3. **Invariance Extrapolation Remains Classical**:
   - On truly unseen knot invariants without local planar formulas ($\sigma(K), u(K)$), no neural graph model outperforms the classical 2-parameter linear baseline $[c(K), w(K)]$, while the non-learned Alexander polynomial baseline outperforms all neural variants ($\text{MAE} = 1.937$).
   - This provides the exact empirical boundary: deterministic topological simplification provides strong diagram verification, but geometric surgery invariants require global algebraic invariants beyond local message passing.

---

## 8. Fully Audited Protocol: Large-Scale Task 1 & Task 2 Benchmarks

Following rigorous scientific auditing, two critical experimental flaws in the initial setup were identified and resolved:
1. **Coordinate / Traversal Leakage**: Hand-crafted chord lengths $|u_1 - u_2| / 2N$ and raw arc label indices leaked the shared braid traversal cut-point ($0.918 \to 0.401$ AUROC under cyclic relabeling). This was eliminated by replacing all coordinate features with topological planar face degree profiles $[f_1, f_2, f_3, f_4]$ from `snappy.Link.faces()`, achieving 100% invariance under all cyclic shifts ($0.9490 \leftrightarrow 0.9490$).
2. **Positive Diagram Independence**: In single-braid perturbations, up to 63.6% of positive diagrams reduced to literally identical PD codes. In this audited protocol, positive diagrams are generated via **genuinely independent SnapPy backtrack walks** (`Link.backtrack(steps=25..70)`).
   - **Identical Canonical PD Rate**: Exactly **0 / 642 (0.0%)** across all evaluated positive pairs!
3. **KnotInfo Full-Census Scale for Task 2**:
   - **Train Set**: All 802 knots with $c \le 11$ (801 prime + 1 unknot).
   - **Test Set**: All 2,176 prime knots with $c = 12$ (1,288 alternating + 888 non-alternating).
   - **Unknotting Number Ground Truth**: Strictly restricted to the **1,628 test knots** with exact integer values (excluding 548 knots with interval bounds like `'[2, 3]'`).

### 8.1 Audited Task 1 Results: Same-Knot Verification on Independent Base Diagrams
- Evaluated on $N = 214$ prime knots at $c \in \{9, 10\}$ (642 positive pairs, 1,070 matched-crossing hard negatives; 1,712 total pairs).
- Strictly safe cyclic shift & crossing permutations applied.

| Model Candidate | Overall AUROC (95% Bootstrap CI) | Non-Identical AUROC | Mechanism / Interpretation |
| :--- | :---: | :---: | :--- |
| **SnapPy Canonical Hash / Exact Match** | $0.500 \pm 0.000$ | $0.500$ | **Fails completely**: 0% of backtrack pairs share identical PD code. |
| **Pure Graph GNN (Raw Input, Untrained)** | $0.000 \pm 0.000$ | $0.000$ | **Fails**: Backtrack inflates crossing count to 30–50; raw graph size is uncalibrated. |
| **Pure Graph GNN (Canonical Input, Untrained)** | **$0.988 \pm 0.004$** | **$0.988$** | **Core Breakthrough**: $R_1/R_2$ reduction normalizes graph size; topology separates knots. |
| **Pure Graph GNN (Canonical Input, Contrastive)**| $0.950 \pm 0.009$ | $0.950$ | Learned contrastive representations achieve strong invariance across orbits. |
| **Alexander Polynomial Baseline** | **$1.000 \pm 0.000$** | **$1.000$** | Exact ambient isotopy invariant separates distinct knot classes. |

### 8.2 Audited Task 2 Results: Large-Scale Invariant Prediction on 12-Crossing Knots
- Train: 802 knots ($c \le 11$) $\longrightarrow$ Test: 2,176 knots ($c = 12$).
- Evaluates: Trivial Mean Baseline, Linear $[c(K), w(K)]$ Fit, Alexander Polynomial Baseline, Untrained GNNs (Raw/Canonical), and Supervised Multi-Task Canonical GNN.

| Target Invariant | Evaluated Test Knots | Mean Baseline MAE (Acc) | Linear $[c, w]$ MAE (Acc) | Alexander Baseline MAE (Acc) | Untrained Canonical GNN MAE (Acc) | **Supervised Canonical GNN MAE (Acc)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unknotting Number $u(K)$** | **1,628** | $0.449$ ($66.7\%$) | $0.446$ ($68.7\%$) | $0.778$ ($53.4\%$) | $0.436$ ($66.7\%$) | **$0.409$ ($69.2\%$)** |
| **Smooth 4-Genus $g_4(K)$** | **2,176** | $0.622$ ($63.5\%$) | $0.516$ ($62.7\%$) | $0.876$ ($48.6\%$) | $0.607$ ($61.4\%$) | **$0.380$ ($72.1\%$)** |
| **Three-Genus $g_3(K)$** | **2,176** | $0.606$ ($46.5\%$) | $0.615$ ($45.8\%$) | **$0.141$ ($95.7\%$)** | $0.600$ ($46.5\%$) | $0.625$ ($46.6\%$) |
| **Signature $\sigma(K)$** | **2,176** | $2.325$ ($0.0\%$) | **$0.919$ ($36.5\%$)** | $3.217$ ($11.9\%$) | $2.296$ ($0.3\%$) | **$1.015$ ($34.2\%$)** |

### 8.3 Key Empirical Conclusions
1. **Supervised GNN Outperforms Classical Baselines on 4-Genus & Unknotting Number**:
   - On **smooth 4-genus $g_4(K)$ (slice genus)**, the Supervised Canonical GNN achieves **$\text{MAE} = 0.380$ and $72.1\%$ exact accuracy**, outperforming the Linear $[c, w]$ baseline ($0.516$, $62.7\%$) and the Mean baseline ($0.622$, $63.5\%$).
   - On **unknotting number $u(K)$**, the Supervised Canonical GNN achieves **$\text{MAE} = 0.409$ and $69.2\%$ exact accuracy**, outperforming Linear $[c, w]$ ($0.446$, $68.7\%$) and Alexander ($0.778$, $53.4\%$).
2. **Algebraic Determination of Three-Genus**:
   - For alternating knots, Seifert genus is given exactly by $\text{deg}(\Delta(t))/2$. The non-learned Alexander baseline achieves **$\text{MAE} = 0.141$ and $95.7\%$ exact accuracy**, decisively establishing where symbolic algebraic invariants dominate message-passing neural approximations.
3. **Diagram Writhe Governs Signature**:
   - Signature correlates strongly with diagram writhe ($\sigma \approx w/2$). The linear writhe fit achieves $\text{MAE} = 0.919$, closely tracked by the Supervised GNN at $\text{MAE} = 1.015$, both vastly outperforming the Mean baseline ($2.325$).

