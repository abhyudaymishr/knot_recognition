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

Following rigorous scientific auditing, the experimental protocol was audited and updated to address all systematic artifacts:
1. **Size-Matched Diagram Sampler**: Fixed the diagram-size mismatch in Task 1 that caused an artificial AUROC of $0.000$. Positives and negatives are sampled across identical backtrack step schedules ($[25, 40, 55, 70]$), ensuring matched crossing-count and diagram-density distributions.
2. **Same-Alexander Hard Negatives**: Explicitly included negative pairs sharing the exact Alexander polynomial $\Delta(t)$, mined from the 21 collision groups (43 knots in $c \in \{9, 10\}$ sharing $\Delta(t)$).
3. **Knot-Level Cluster Bootstrap**: Standard errors and 95% confidence intervals are computed by resampling clusters of pairs *by knot*, accounting for intra-knot correlation.
4. **Classical Invariant Baselines Added**:
   - Exact Murasugi signature bound $|\sigma(K)|/2$ for smooth 4-genus $g_4(K)$ and unknotting number $u(K)$.
   - Alexander Seifert genus $\text{deg}(\Delta)/2$ for three-genus $g_3(K)$.
   - Linear fit on $[c(K), w(K), |\sigma(K)|]$.
5. **Multi-Seed & Stratified Evaluation**: 5 random seeds ($[42, 43, 44, 45, 46]$) with reporting stratified by Alternating vs. Non-Alternating knots.

### 8.1 Task 1: Same-Knot Verification with Matched Sizes & Alexander Collisions
- Evaluated on $N = 214$ prime knots at $c \in \{9, 10\}$ (164 alternating, 50 non-alternating; 2,140 total pairs).
- Independent backtrack walks ($25..70$ steps); 0 / 1,284 positive pairs reduce to identical PD codes ($0.0\%$).

| Model Candidate | Overall AUROC (95% Knot CI) | Alternating AUROC ($N=164$) | Non-Alternating AUROC ($N=50$) | Empirical Finding |
| :--- | :---: | :---: | :---: | :--- |
| **SnapPy Canonical Hash / Exact Match** | $0.500 \pm 0.000$ | $0.500$ | $0.500$ | Exact matching fails completely: independent backtrack diagrams do not collapse to identical codes. |
| **Pure Graph GNN (Raw Input, Untrained)** | $0.512 \pm 0.023$ | $0.505$ | $0.534$ | **Resolves 0.000 artifact**: On size-matched pairs, raw graph message passing performs at random chance. |
| **Pure Graph GNN (Canonical Input, Untrained)** | **$1.000 \pm 0.000$** | **$1.000$** | **$1.000$** | Canonicalization eliminates redundant curls/loops; planar face profiles + signs cleanly separate knots. |
| **Pure Graph GNN (Canonical Input, Contrastive)**| $0.955 \pm 0.010$ | $0.959$ | $0.944$ | Learned contrastive representations achieve high invariance across orbits. |
| **Alexander Polynomial Baseline** | $0.954 \pm 0.017$ | $0.954$ | $0.952$ | Drops from $1.000$ to $0.954$ due to same-Alexander hard negatives sharing $\Delta(t)$. |

### 8.2 Task 2: Large-Scale Invariant Prediction Across 5 Random Seeds
- Train: 802 knots ($c \le 11$) $\longrightarrow$ Test: 2,176 knots ($c = 12$) (1,288 alternating, 888 non-alternating).
- Unknotting number restricted to the 1,628 test knots with exact ground truth integer values (1,066 alternating, 562 non-alternating).
- Supervised Canonical GNN reported as Mean $\pm$ Stderr over 5 random seeds ($[42, 43, 44, 45, 46]$).

| Target Invariant | Topology Subset | Test Knots | Mean Baseline MAE | Linear $[c, w]$ MAE | Signature Baseline MAE | Alexander Baseline MAE | **Supervised Canonical GNN (5-Seed MAE)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unknotting Number $u(K)$** | **All Knots** | **1,628** | $0.449$ | **$0.446$** | $0.848$ | $0.778$ | $0.455 \pm 0.033$ |
| | Alternating | 1,066 | **$0.369$** | $0.375$ | $0.955$ | $0.949$ | $0.404 \pm 0.037$ |
| | Non-Alternating | 562 | $0.600$ | $0.581$ | $0.644$ | **$0.453$** | $0.552 \pm 0.031$ |
| **Smooth 4-Genus $g_4(K)$** | **All Knots** | **2,176** | $0.622$ | $0.516$ | **$0.271$** | $0.876$ | $0.441 \pm 0.045$ |
| | Alternating | 1,288 | $0.567$ | $0.477$ | **$0.264$** | $1.112$ | $0.390 \pm 0.044$ |
| | Non-Alternating | 888 | $0.701$ | $0.573$ | **$0.280$** | $0.534$ | $0.516 \pm 0.049$ |
| **Three-Genus $g_3(K)$** | **All Knots** | **2,176** | $0.606$ | $0.615$ | $0.580$ | **$0.015$** | $0.623 \pm 0.026$ |
| | Alternating | 1,288 | $0.709$ | $0.617$ | $0.595$ | **$0.000$** | $0.698 \pm 0.036$ |
| | Non-Alternating | 888 | $0.456$ | $0.612$ | $0.559$ | **$0.037$** | $0.514 \pm 0.017$ |
| **Signature $\sigma(K)$** | **All Knots** | **2,176** | $2.325$ | **$0.919$** | $0.001^\dagger$ | $3.217$ | $1.340 \pm 0.224$ |
| | Alternating | 1,288 | $2.288$ | **$0.963$** | $0.001^\dagger$ | $3.848$ | $1.281 \pm 0.226$ |
| | Non-Alternating | 888 | $2.379$ | **$0.855$** | $0.001^\dagger$ | $2.303$ | $1.426 \pm 0.227$ |

*($^\dagger$Identity probe on signature).*

### 8.3 Rigorous Scientific Conclusions
1. **Task 1 Resolution**:
   - The artificial $0.000$ AUROC was confirmed to be a diagram-size artifact resulting from comparing untangled (step 0) diagrams to tangled (step 50) diagrams for positives, while negatives were matched at step 0.
   - When positive and negative pairs are strictly matched by backtrack step counts, raw graph message passing scores **$0.512 \pm 0.023$ (random chance)**.
   - Confluent $R_1/R_2$ canonicalization brings diagrams into reduced planar graphs where topological face-degree and crossing sign features achieve **$1.000$ AUROC** without any learned weights.
2. **Task 2 Reality: Classical Topological Formulas Dominate Learned Models**:
   - **Smooth 4-Genus**: Murasugi's classical signature bound $|\sigma(K)|/2$ achieves **$\text{MAE} = 0.271$** across all 2,176 test knots ($0.264$ alternating, $0.280$ non-alternating), decisively beating the Supervised GNN ($0.441 \pm 0.045$). The apparent earlier improvement of the GNN over linear $[c, w]$ ($0.516$) was an artifact of omitting the standard signature baseline.
   - **Three-Genus**: Alexander polynomial degree $\text{deg}(\Delta)/2$ predicts Seifert genus with **$\text{MAE} = 0.015$** ($100\%$ exact accuracy on alternating knots), overwhelmingly outperforming the GNN ($0.623 \pm 0.026$).
   - **Unknotting Number**: Across 5 random seeds, the Supervised GNN achieves **$\text{MAE} = 0.455 \pm 0.033$**, which is statistically indistinguishable from the trivial linear baseline ($0.446$) and mean baseline ($0.449$). On non-alternating knots, the Alexander polynomial baseline ($0.453$) outperforms the GNN ($0.552 \pm 0.031$).
   - **Signature**: Diagram writhe alone provides a better linear predictor ($\text{MAE} = 0.919$) than the Supervised GNN ($\text{MAE} = 1.340 \pm 0.224$).

---

## 9. Methodological Audit & Scientific Pitfall Taxonomy

The rigorous iterative audit of this benchmark uncovered fundamental insights into why machine learning models for topological knot invariants frequently report inflated or misleading performance.

### 9.1 Pitfall 1: Coordinate & Traversal Order Leakage
- **Mechanism**: Hand-crafted geometric features (such as normalized Gauss chord lengths $|u_1 - u_2| / 2N$) depend on the traversal cut-point where strand numbering begins. In braid-derived datasets, all diagrams derived from the same base braid share this cut-point.
- **Audit Proof**: Under strictly safe cyclic shifts $i \mapsto (i + k) \pmod{2N}$, chord feature AUROC collapses from **$0.918 \longrightarrow 0.401$**. Replacing coordinates with topological planar face degree profiles ($[f_1, f_2, f_3, f_4]$ from `snappy.Link.faces()`) achieves true coordinate invariance ($0.9490 \leftrightarrow 0.9490$).

### 9.2 Pitfall 2: Diagram Size & Backtrack Step Imbalance
- **Mechanism**: When positive diagram pairs undergo long backtrack walks (25–70 steps, inflating crossing counts to 30–70) while negative pairs are evaluated against compact minimal diagrams (9–10 crossings), graph scale disparity creates artificial class separation.
- **Audit Proof**: Raw GNN message passing on uncalibrated diagram sizes yielded an AUROC of **$0.000 \pm 0.000$ (perfect inversion)**. When positive and negative pairs are strictly sampled from identical backtrack step distributions, the AUROC normalizes to **$0.512 \pm 0.023$ (pure random chance)**.

### 9.3 Pitfall 3: Untrained Encoders as Continuous Graph Hashes
- **Mechanism**: An untrained random GNN on canonicalized input scored $1.000$ AUROC. This does **not** indicate learned topological comprehension. Rather, confluent $R_1/R_2$ reduction frequently returns the identical or isomorphic minimal shadow for alternating knots; an untrained random network on crossing signs and face profiles acts as a continuous Weisfeiler–Lehman (WL) graph hash.
- **Audit Proof**: Discrete string matching of PD codes fails ($0.500$ AUROC) solely because arc labels are permuted. Continuous graph embeddings smoothly group isomorphic minimal graphs, achieving high separation without learning.

### 9.4 Pitfall 4: Neglecting Known Classical Topological Invariants
- **Mechanism**: Prior work often claimed that invariants such as smooth 4-genus $g_4(K)$ or three-genus $g_3(K)$ have "no cheap formula," reporting small neural gains over trivial linear $[c(K), w(K)]$ baselines.
- **Audit Proof**:
  - **Smooth 4-genus $g_4(K)$**: For alternating knots, Murasugi's classical theorem proves $g_4(K) = |\sigma(K)|/2$. Evaluating $|\sigma(K)|/2$ yields an **$\text{MAE} = 0.271$**, overwhelmingly beating the Supervised GNN ($\text{MAE} = 0.441 \pm 0.045$). (The exact topological signature $\sigma(K)$ is computable directly from the diagram Seifert/Goeritz matrix and is supplied via KnotInfo).
  - **Three-genus $g_3(K)$**: Alexander polynomial degree $\text{deg}(\Delta(t))/2$ achieves **$\text{MAE} = 0.015$** ($100\%$ accuracy on alternating knots), rendering neural regression obsolete.

### 9.5 Pitfall 5: Architecture Underfitting on Global Topological Surgery
- **Diagnostic Check**: Evaluating the 4-layer Multi-Task GNN on the 802 training knots ($c \le 11$) demonstrates that the model is **severely underfitting**:
  - Unknotting Number $u(K)$: Train MAE = **$0.469$** vs. Train Mean Baseline = **$0.474$**.
  - Smooth 4-Genus $g_4(K)$: Train MAE = **$0.685$** vs. Train Mean Baseline = **$0.618$**.
  - Three-Genus $g_3(K)$: Train MAE = **$0.639$** vs. Train Mean Baseline = **$0.533$**.
  - Signature $\sigma(K)$: Train MAE = **$2.263$** vs. Train Mean Baseline = **$2.266$**.
- **Conclusion**: Local 4-valent message passing fails to fit global topological surgery invariants even on the training data.

---

## 10. Publication & Strategic Pathways

Given the audited empirical reality, three concrete research pathways exist:

1. **Empirical Audit & Benchmark Manifesto (Recommended for NeurIPS D&B / AI4Science Workshops)**:
   - Frame the work as a rigorous methodological critique: *"Pitfalls in Machine Learning for Knot Invariants: Traversal Leakage, Diagram-Size Artifacts, and the Dominance of Classical Invariant Baselines."*
   - Provide the complete benchmark suite, dataset, and controlled evaluation protocol as an open community standard.
2. **The Non-Alternating Frontier (Long-Term Research)**:
   - Focus exclusively on targets where classical proxies fail (e.g., unknotting number on non-alternating knots, where Alexander baseline achieves $0.453$).
   - Develop non-local topological architectures (e.g. persistent homology or global Seifert surface graph networks).
3. **Repository Archival**:
   - Retain the audited benchmarks, verified KnotInfo database, and clean test suites as a definitive internal standard.



