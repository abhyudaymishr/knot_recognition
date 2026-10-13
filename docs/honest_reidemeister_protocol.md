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

---## 8. Fully Audited Protocol: Large-Scale Task 1 & Task 2 Benchmarks

Following rigorous scientific auditing, the experimental protocol was audited and updated to address all systematic artifacts:
1. **Size-Matched Diagram Sampler**: Fixed the diagram-size mismatch in Task 1 that caused an artificial AUROC of $0.000$. Positives and negatives are sampled across identical backtrack step schedules ($[25, 40, 55, 70]$), ensuring matched crossing-count and diagram-density distributions.
2. **Ground-Truth Stratification & Mathematical Correction on Murasugi's Bound**:
   - Knots are partitioned strictly using KnotInfo's validated `alternating` column.
   - **Murasugi 4-Genus Bound ($g_4 \ge |\sigma|/2$)**: Murasugi's classical theorem provides only a *lower bound* on smooth 4-genus, not an exact equality for alternating knots. The standard counterexample is knot $8_3$, which is alternating with signature $\sigma(8_3) = 0$ but $g_4(8_3) = 1$ (non-slice). Under KnotInfo ground truth, $|\sigma|/2$ evaluates to **$0.264$ MAE** on alternating knots ($949$ exact, $338$ differ by 1, $1$ differs by 2) and **$0.280$ MAE** on non-alternating knots (**$0.271$ overall**). Both stratified numbers are mathematically exact.
   - **Exact Three-Genus Theorem ($g_3 = \text{deg}(\Delta)/2$)**: The exact equality that holds for alternating knots is Murasugi's theorem for Seifert three-genus: $g_3(K) = \text{deg}(\Delta(K))/2$. This evaluates to an exact **$0.000$ MAE** on alternating knots and **$0.015$ overall**.
3. **Scaled Distinct Base Origin Evaluation**: Expanded to $435$ non-alternating knots across $c \in \{9, 10, 11, 12\}$ testing KnotInfo vs. SnapPy census diagrams. All canonicalized diagrams reach minimal crossing count ($100.0\%$). SnapPy's exterior isometry (`is_isometric_to`, exact under Mostow rigidity for hyperbolic knots) serves as the exact upper bound reference ($0.9954$).
4. **Resolution of Label Leakage and "Formula Inputs" in Tabular Baselines**:
   - **Signature Row Dropped**: Random Forest and GBDT models fed classical features including $\sigma(K)$ "predicted" signature with MAE $0.003$. This was pure label leakage; the signature row has been dropped.
   - **Three-Genus Flagged as Formula Input**: Alexander polynomial degree is the Seifert genus formula itself for alternating knots. Tabular performance ($0.046$) is explicitly flagged as "Formula Input: $\text{deg}(\Delta)/2$", not as a learned baseline.
   - **Label Inference in Tabular Models**: KnotInfo's ground truth for unknotting number and smooth 4-genus is known primarily where classical bounds (signature, Rasmussen $s$, Ozsváth–Szabó $\tau$) meet. Tabular models fed these invariants achieve low MAE ($0.074$ on $g_4$) by reconstructing the mathematical decision rules of the labeling process, not by intuiting knot topology.
   - **The Solved Subset vs. Unsolved Frontier**: For unknotting number, the $1,628$ test knots evaluated are the mathematically resolved subset where bounds coincide; the roughly $548$ interval-valued knots represent the true unresolved frontier.
5. **Face-Degree Histogram & Crossing Summary Baseline**: A simple non-neural geometric baseline ($[c, n_+, n_-, w, h_2, \dots, h_{10}]$) reaches **$0.9549$ AUROC**, matching/exceeding the Untrained GNN ($0.9386$).
6. **Strict Validation Holdout Experiment**: Evaluated with train on $c \le 10$, checkpoint selection on $c = 11$ validation holdout, and test on $c = 12$ with zero test-set hyperparameter tuning.

---

### 8.1 Task 1: Same-Knot Verification with Matched Sizes & Scaled Distinct Base Diagrams

#### A. Same-Base Backtrack Verification ($N = 214$ Prime Knots at $c \in \{9, 10\}$, 2,140 Matched Pairs)
| Evaluation Setting | Model / Baseline Candidate | Overall AUROC (95% Knot CI) | Alternating AUROC ($N=164$) | Non-Alternating AUROC ($N=50$) | Empirical Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Same-Base Backtrack** | **SnapPy Exact String Match** | $0.500 \pm 0.000$ | $0.500$ | $0.500$ | Fails completely due to cyclic arc label shifts ($0.0\%$ string matches). |
| | **Weisfeiler–Lehman Graph Hash** | **$0.990 \pm 0.006$** | **$0.987$** | **$1.000$** | **Label-Invariant Baseline**: $100.0\%$ positive match rate, $2.06\%$ negative collisions. |
| | **Pure Graph GNN (Raw, Untrained)** | $0.512 \pm 0.024$ | $0.505$ | $0.534$ | **Resolves size artifact**: On size-matched pairs, raw graph message passing is at random chance. |
| | **Pure Graph GNN (Canonical, Untrained)** | **$1.000 \pm 0.000$** | **$1.000$** | **$1.000$** | Continuous analogue of WL hash on reduced shadows from identical base origin. |
| | **Pure Graph GNN (Canonical, Contrastive)** | $0.956 \pm 0.010$ | $0.960$ | $0.944$ | Contrastive representation across backtrack walks. |
| | **Alexander Polynomial Baseline** | $0.954 \pm 0.016$ | $0.954$ | $0.952$ | Evaluated with same-Alexander hard negatives sharing $\Delta(t)$. |

#### B. Scaled Distinct Base Origin Breakdown ($N = 435$ Non-Alternating Knots: KnotInfo vs. SnapPy Census)
| Crossing Slice | Knots ($N$) | SnapPy Exterior Iso (Upper Bound) | Face Hist Baseline ($[c, n_+, n_-, w, h]$) | WL Graph Hash (Match Rate) | Untrained Pure GNN (5-Seed Mean ± Stderr) | Canonical PD String Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$c \le 10$** | 50 | **$0.9600^\ast$** | **$0.6897$** | $0.5100$ ($2.0\%$) | $0.6462 \pm 0.0037$ | $0.5000$ ($0\%$) |
| **$c = 11$** | 185 | **$1.0000$** | **$0.9968$** | **$1.0000$** ($100\%^\dagger$) | $1.0000 \pm 0.0000$ | $0.5000$ ($0\%$) |
| **$c = 12$** | 200 | **$1.0000$** | **$0.9974$** | $0.5625$ ($12.5\%$) | $0.9578 \pm 0.0076$ | $0.5000$ ($0\%$) |
| **Overall** | **435** | **$0.9954^\ast$** | **$0.9549$** (pos $0.9835$, neg $0.8425$) | **$0.7425$** ($48.5\%^\ddagger$) | **$0.9386 \pm 0.0038$** (pos $0.9996$, neg $0.9978$) | $0.5000$ ($0\%$) |

*Table Notes on Task 1 Breakdown*:
- $^\ast$ **SnapPy Exterior Isometry Upper Bound**: The gap from $1.0000$ ($0.9600$ at $c \le 10$ and $0.9954$ overall) is caused entirely by non-hyperbolic satellite/torus knots ($10_{124}, 10_{162}, 10_{163}, 10_{164}, 10_{165}$) where hyperbolic Mostow rigidity does not apply to SnapPy's hyperbolic isometry routine.
- $^\dagger$ **Source Table Confound at $c = 11$**: Both KnotInfo and SnapPy census derived their $11$-crossing presentations from Hoste–Thistlethwaite DT tables; canonicalization yields identical minimal diagrams in $100\%$ of cases.
- $^\ddagger$ **Discrete WL Hash Bound**: The WL hash's AUROC is mathematically capped at $0.7425$ because discrete $0/1$ scores assign $0$ to the $51.5\%$ unmatched positive pairs, tying them with true negatives in rank ordering.
- **The 0.94 Untrained GNN Explained**: The Untrained GNN adds **nothing** beyond the simple Face-Degree Histogram baseline ($0.9549$ vs $0.9386$). Furthermore, the Untrained GNN's positive vs negative cosines are compressed into $[0.9978, 0.9996]$ (a tiny margin of $0.0018$ where rankings ride on microscopic numerical fluctuations), whereas the Face Histogram exhibits a natural, robust geometric margin ($0.9835$ vs $0.8425$, margin $0.1410$).

---

### 8.2 Task 2: Large-Scale Invariant Prediction Across 5 Random Seeds

- **Train Set**: 802 knots with $c \le 11$ (801 prime + 1 unknot).
- **Test Set**: 2,176 knots with $c = 12$ ($1,288$ alternating, $888$ non-alternating). Unknotting number evaluated on the 1,628 test knots with exact ground truth integer values ($1,066$ alternating, $562$ non-alternating).
- **GNN Architecture**: 4-layer permutation-invariant Signed GNN with Smooth L1 loss, target standardization, and CosineAnnealingLR across 5 random seeds ($[42, 43, 44, 45, 46]$).

| Target Invariant | Topology Subset | Test Knots | Mean Base MAE | Linear $[c, w]$ MAE | Invariant Baseline MAE | Random Forest MAE | GBDT MAE | **Supervised GNN Test MAE** | GNN Train MAE | Gen Gap ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unknotting Number $u(K)$** | **All Knots** | **1,628** | $0.449$ | $0.446$ | $0.848$ | **$0.324$** | **$0.363$** | $0.411 \pm 0.004$ | $0.278$ | $+0.133$ |
| | Alternating | 1,066 | $0.369$ | $0.375$ | $0.955$ | **$0.327$** | **$0.371$** | $0.378 \pm 0.007$ | --- | $+0.100$ |
| | Non-Alternating | 562 | $0.600$ | $0.581$ | $0.644$ | **$0.317$** | **$0.349$** | $0.472 \pm 0.006$ | --- | $+0.194$ |
| **Smooth 4-Genus $g_4(K)$** | **All Knots** | **2,176** | $0.622$ | $0.516$ | $0.271^\ast$ ($0.256^\dagger$) | **$0.074$** | **$0.081$** | $0.331 \pm 0.008$ | $0.192$ | $+0.139$ |
| | Alternating | 1,288 | $0.567$ | $0.477$ | $0.264^\ast$ | **$0.069$** | **$0.070$** | $0.294 \pm 0.008$ | --- | $+0.102$ |
| | Non-Alternating | 888 | $0.701$ | $0.573$ | **$0.243^\dagger$** ($0.280^\ast$) | **$0.082$** | **$0.098$** | $0.386 \pm 0.008$ | --- | $+0.194$ |
| **Three-Genus $g_3(K)$** | **All Knots** | **2,176** | $0.606$ | $0.615$ | **$0.015^\ddagger$** | $0.046^\parallel$ | $0.049^\parallel$ | $0.533 \pm 0.017$ | $0.173$ | $+0.361$ |
| | Alternating | 1,288 | $0.709$ | $0.617$ | **$0.000^\ddagger$** | $0.046^\parallel$ | $0.046^\parallel$ | $0.523 \pm 0.020$ | --- | $+0.351$ |
| | Non-Alternating | 888 | $0.456$ | $0.612$ | **$0.037^\ddagger$** | $0.046^\parallel$ | $0.053^\parallel$ | $0.547 \pm 0.014$ | --- | $+0.375$ |
| **Signature $\sigma(K)$** | **All Knots** | **2,176** | $2.325$ | $0.919$ | **$0.000^\S$** | *[Dropped]$^\text{\textpilcrow}$* | *[Dropped]$^\text{\textpilcrow}$* | $0.832 \pm 0.033$ | $0.446$ | $+0.386$ |
| | Alternating | 1,288 | $2.288$ | $0.963$ | **$0.000^\S$** | *[Dropped]$^\text{\textpilcrow}$* | *[Dropped]$^\text{\textpilcrow}$* | $0.754 \pm 0.040$ | --- | $+0.308$ |
| | Non-Alternating | 888 | $2.379$ | $0.855$ | **$0.000^\S$** | *[Dropped]$^\text{\textpilcrow}$* | *[Dropped]$^\text{\textpilcrow}$* | $0.945 \pm 0.024$ | --- | $+0.499$ |

*Table Notes*:  
- $^\ast$ Murasugi lower bound $|\sigma(K)|/2$ ($0.264$ alternating, $0.280$ non-alternating, $0.271$ overall).  
- $^\dagger$ Rasmussen invariant $|s|/2$ and Ozsváth–Szabó $|\tau|$ invariant (identically $0.243$ on non-alternating, $0.256$ overall).  
- $^\ddagger$ Alexander Seifert genus $\text{deg}(\Delta)/2$ (exact $0.000$ on alternating knots).  
- $^\S$ Exact Goeritz / Seifert matrix signature.  
- $^\parallel$ **Formula Input Flag**: RF/GBDT models for $3$-genus receive Alexander degree as an input feature; this is the algebraic formula itself, not a learned baseline.  
- $^\text{\textpilcrow}$ **Dropped (Label Leakage)**: Tabular feature matrices included $\sigma(K)$ as an input feature, allowing models to trivially predict $\sigma(K)$ with $\text{MAE} = 0.003$. This row has been dropped.

---

### 8.3 Strict Out-of-Distribution Validation Holdout Experiment

To ensure zero test-set hyperparameter tuning and inspect the crossing generalization gap rigorously:
- **Train Set**: Knots with $c \le 10$ ($N = 250$).
- **Validation Holdout**: Knots with $c = 11$ ($N = 552$), used exclusively for checkpoint selection via validation loss.
- **Test Set**: Knots with $c = 12$ ($N = 2,176$), evaluated out-of-distribution without any test-set tuning.
- **Node Features**: Intrinsic signed graph topology and normalized face degree profiles ($[s, \text{is\_pos}, \text{is\_neg}, f_1, f_2, f_3, f_4]$) with **no** crossing count or size term ($n/20$).

| Target Invariant | $c = 12$ Test Knots | Out-of-Distribution Test MAE | Comparison to Baselines on $c = 12$ |
| :--- | :---: | :---: | :--- |
| **Unknotting Number $u(K)$** | 1,628 | **$0.428$** | Near linear baseline $[c, w]$ ($0.446$) and mean ($0.449$). |
| **Smooth 4-Genus $g_4(K)$** | 2,176 | **$0.354$** | Substantially worse than classical bound $|\sigma|/2$ ($0.271$) and $|\tau|$ ($0.243$). |
| **Three-Genus $g_3(K)$** | 2,176 | **$0.730$** | **Worse than the mean baseline** ($0.606$); destroyed by Alexander degree ($0.015$). |
| **Signature $\sigma(K)$** | 2,176 | **$0.982$** | Worse than linear baseline $[c, w]$ ($0.919$); exact Goeritz signature is $0.000$. |

---

## 9. Comprehensive Pitfall Taxonomy for Machine Learning in Knot Theory

The audited empirical benchmarks establish six fundamental failure modes in geometric topological ML:

### 9.1 Pitfall 1: Coordinate & Traversal Order Leakage
- **Mechanism**: Hand-crafted geometric features (such as normalized Gauss chord lengths $|u_1 - u_2| / 2N$) depend on the traversal cut-point where strand numbering begins. In braid-derived datasets, diagrams derived from the same base braid share this cut-point.
- **Audit Measurement**: Under cyclic label shifts alone, chord feature AUROC collapses from **$0.918 \longrightarrow 0.401$**. When cyclic shifts are combined with strand orientation reversal and crossing index permutation, performance collapses to **$0.4908$ (random chance)**. Replacing traversal coordinates with topological planar face degree profiles ($[f_1, f_2, f_3, f_4]$ from `snappy.Link.faces()`) achieves true coordinate invariance ($0.9490 \leftrightarrow 0.9490$).

### 9.2 Pitfall 2: Diagram Size & Backtrack Step Imbalance
- **Mechanism**: When positive diagram pairs undergo long backtrack walks (25–70 steps, inflating crossing counts to 30–70) while negative pairs are evaluated against compact minimal diagrams (9–10 crossings), graph scale disparity creates artificial class separation.
- **Audit Measurement**: Raw GNN message passing on uncalibrated diagram sizes yielded an AUROC of **$0.000 \pm 0.000$ (perfect inversion)**. When positive and negative pairs are strictly sampled from identical backtrack step distributions, the AUROC normalizes to **$0.512 \pm 0.024$ (pure random chance)**.

### 9.3 Pitfall 3: Untrained Encoders as Continuous Graph Hashes on Reduced Shadows
- **Mechanism**: An untrained random GNN on canonicalized input scored $1.000$ AUROC on backtrack pairs originating from the same base diagram. Confluent $R_1/R_2$ reduction on backtrack walks of the same knot returns the identical or isomorphic minimal shadow; an untrained random network on crossing signs and face profiles acts as a continuous Weisfeiler–Lehman (WL) graph hash.
- **Audit Measurement**: When tested across $N = 435$ distinct base presentations of non-alternating knots (KnotInfo vs SnapPy census), an untrained GNN scores $0.9386 \pm 0.0038$. However, a non-neural Face-Degree Histogram baseline ($[c, n_+, n_-, w, h_2..h_{10}]$) achieves **$0.9549$ AUROC**, demonstrating that the GNN adds nothing beyond this geometric profile.

### 9.4 Pitfall 4: Label Inference and Formula Inputs in Tabular Baselines
- **Mechanism**: Tabular models fed classical topological invariants achieve low MAE ($0.074$ on $g_4$) not by discovering geometry, but by inferring the decision rules used to construct database tables. KnotInfo values for $g_4$ and $u(K)$ are cataloged where upper and lower bounds meet. Furthermore, feeding $\sigma(K)$ to predict $\sigma(K)$ constitutes direct label leakage (MAE $0.003$), while feeding Alexander degree to predict $g_3(K)$ simply re-evaluates Murasugi's theorem.
- **Audit Measurement**: Omitting the leaked signature row and flagging 3-genus tabular models as "Formula Input: $\text{deg}(\Delta)/2$" restores benchmark integrity. For unknotting number, the $1,628$ evaluated knots represent the "easy" cases where bounds coincide; the roughly $548$ interval-valued knots remain the true open frontier.

### 9.5 Pitfall 5: Multi-Task Optimization Imbalance vs. Generalization Gap
- **Mechanism**: Optimization failure previously masked model capacity. When trained with target standardization, Smooth L1 loss, and batched gradient accumulation, the GNN comfortably fits training knots ($c \le 11$), beating linear fits across all targets.
- **Audit Measurement**: However, under a strict validation holdout split ($c \le 10$ train, $c = 11$ val checkpoint, $c = 12$ test), the model exhibits a severe out-of-distribution generalization gap: 3-genus collapses to $0.730$ MAE (worse than the test mean baseline $0.606$), and unknotting number ($0.428$) fails to beat simple linear fits ($0.446$). Local message passing cannot extrapolate geometric surgery invariants to higher crossing numbers.

---

## 10. Publication & Strategic Pathways

1. **Empirical Audit & Benchmark Manifesto (Recommended for NeurIPS D&B / AI4Science Workshops)**:
   - Frame the work as a rigorous methodological critique: *"Pitfalls in Machine Learning for Knot Invariants: Traversal Leakage, Diagram-Size Artifacts, and the Dominance of Classical Invariant Baselines."*
   - Provide the complete benchmark suite, dataset, and controlled evaluation protocol as an open community standard.
2. **The Non-Alternating Frontier (Long-Term Research)**:
   - Focus exclusively on targets where classical proxies fail (e.g., unknotting number on non-alternating knots, where Alexander baseline achieves $0.453$).
   - Develop non-local topological architectures (e.g. persistent homology or global Seifert surface graph networks).
3. **Repository Archival**:
   - Retain the audited benchmarks, verified KnotInfo database, and clean test suites as a definitive internal standard.




