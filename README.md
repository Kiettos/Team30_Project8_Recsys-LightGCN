# Team 30 — Project 8: LightGCN vs Matrix Factorization (MovieLens-1M)

A comparison of **LightGCN** (Graph Convolutional Network for collaborative filtering) against the **BPR-MF** (Matrix Factorization) baseline on MovieLens-1M, investigating the impact of **embedding size** and **number of propagation layers**, accompanied by an **interactive user–item graph visualization demo + cold-start recommendation for new users**.

| | |
|---|---|
| Dataset | MovieLens-1M → 6,034 users, 3,125 movies, 574,376 interactions |
| Metric | Recall@20, NDCG@20 (full-ranking, masking known items) |
| Split | **Temporal** (last 10% = test, preceding 10% = validation) |
| Main Model | LightGCN (BPR loss, layer-averaged embeddings) |
| Baseline | BPR-MF (0 propagation layers) |
| TEST Results (3 seeds) | LightGCN **0.1226 ± 0.0018** / **0.0781 ± 0.0012** · MF **0.1194 ± 0.0004** / **0.0764 ± 0.0006** |

---

## 1. Team Members

| Member | Student ID | Responsibilities |
|---|---|---|
| Lương Chí Kiệt | *(to be filled)* | Preprocessing, LightGCN, HPO, evaluation, demo, report |
| *(Member 2)* | *(to be filled)* | *(to be filled)* |

## 2. Problem Statement and Dataset

**Problem Statement.** For each user, rank all unwatched movies and retrieve the top-20. The dataset is treated as **implicit feedback**: any rating ≥ 4 is considered a positive interaction (lower ratings are discarded as they do not reflect user preference).

**Preprocessing** (`src/data/preprocess_ml1m.py`):

| Step | Result |
|---|---|
| Read `data/ratings.dat` | 1,000,209 ratings · 6,040 users · 3,706 movies |
| Filter `rating ≥ 4` (implicit) | 575,281 interactions |
| Deduplicate + 5-core filtering (all users & movies ≥ 5 interactions) | **574,376 interactions · 6,034 users · 3,125 movies · 3.05% density** |
| Temporal split per user (10% test, 10% val) | **459,436 train / 57,470 val / 57,470 test** |
| Bipartite graph for LightGCN | 9,159 × 9,159, 918,872 non-zero entries |

**Why temporal split?** Random splitting leaks future information ("future" movies can end up in the training set), leading to artificially inflated scores. Here, within each user's history sorted by timestamp, the last 10% goes to test, the preceding 10% to validation, and the remainder to train. The script also cross-checks that no `(user, item)` pair appears across multiple splits, and every user satisfies `max(ts train) ≤ min(ts val) ≤ max(ts val) ≤ min(ts test)`.

## 3. Methodology

### 3.1 Baseline: BPR-MF

`ŝ(u,i) = p_u · q_i`, optimized with BPR loss:

$$\mathcal{L} = -\log \sigma(\hat s_{u,i^+} - \hat s_{u,i^-}) + \lambda \left(\Vert{}p_u\Vert{}^2 + \Vert{}q_{i^+}\Vert{}^2 + \Vert{}q_{i^-}\Vert{}^2\right)$$

Negative items are sampled uniformly and resampled if they hit a movie already watched by the user. MF is equivalent to LightGCN with **0 propagation layers**, meaning both models share the exact same dataset, evaluation functions, and hyperparameter space for a fair comparison.

### 3.2 Main Model: LightGCN

Layer propagation is performed on the user–item bipartite graph using only training edges:

$$e_u^{(k+1)} = \sum_{i \in \mathcal{N}(u)} \frac{1}{\sqrt{d_u\, d_i}}\, e_i^{(k)}, \qquad e_i^{(k+1)} = \sum_{u \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i\, d_u}}\, e_u^{(k)}$$

Final embedding = **mean across all layers** (including layer 0), predicted score `ŝ(u,i) = e_u · e_i`, with the same BPR loss and L2 regularization on initial embeddings. No feature transformation or nonlinear dropout is used—staying true to the LightGCN design. The implementation uses sparse tensors to avoid constructing a dense 9,159² matrix.

### 3.3 Evaluation Protocol

- **Full ranking** over all 3,125 movies, masking items already known to the user (train + val when evaluating test).
- Recall@20 = `|top20 ∩ target| / |target|`; NDCG@20 with IDCG truncated at `min(|target|, 20)`.
- **HPO selects solely based on validation.** The test set is evaluated **only once** in `--final` across **3 seeds** (0, 1, 2), reporting **mean ± std**—without cherry-picking the best run out of 3.

## 4. Experimental Setup

| | MF | LightGCN |
|---|---|---|
| Number of HPO configs | 12 | 11 |
| Search grid | dim {32, 64, 128, 256}; lr × reg | layers {1,2,3,4}; dim {32,64,128}; lr × reg |
| Embedding dim | 32–256 | 32–128 |
| Batch size / max epochs / patience | 2048 / 300 / 10 | 2048 / 300 / 10 |
| Early stopping | Based on VAL NDCG@20 (min_delta 1e-4) | Same as MF |
| Final | 3 seeds × best config | Same as MF |

Hardware: **RTX 3050 Laptop 6 GB**, PyTorch 2.5.1 + CUDA 12.1, Python 3.11.
Measured runtime: MF ~2.2 s/epoch (dim 256), LightGCN ~9 s/epoch (dim 128, 2 layers).
Total HPO time: MF ~23 mins, LightGCN ~1 hr 55 mins; `--final` (3 seeds): MF ~7 mins, LightGCN ~55 mins.

## 5. Results

### 5.1 HPO — Matrix Factorization (selected by validation)

| Config | dim | lr | reg | Best epoch | VAL Recall@20 | VAL NDCG@20 |
|---|---|---|---|---|---|---|
| A_mf_d32 | 32 | 0.001 | 0.0001 | 52 | 0.1393 | 0.0909 |
| A_mf_d64 | 64 | 0.001 | 0.0001 | 56 | 0.1424 | 0.0908 |
| A_mf_d128 | 128 | 0.001 | 0.0001 | 50 | 0.1451 | 0.0929 |
| A_mf_d256 | 256 | 0.001 | 0.0001 | 48 | 0.1448 | 0.0929 |
| B_mf_lr0.001_reg0.0003 | 64 | 0.001 | 0.0003 | 56 | 0.1436 | 0.0917 |
| B_mf_lr0.001_reg0.001 | 64 | 0.001 | 0.001 | 56 | 0.1454 | 0.0932 |
| B_mf_lr0.003_reg0.0001 | 64 | 0.003 | 0.0001 | 24 | 0.1368 | 0.0880 |
| B_mf_lr0.003_reg0.0003 | 64 | 0.003 | 0.0003 | 24 | 0.1372 | 0.0888 |
| B_mf_lr0.003_reg0.001 | 64 | 0.003 | 0.001 | 49 | 0.1466 | 0.0938 |
| C_mf_d128_lr0.001_reg0.001 | 128 | 0.001 | 0.001 | 50 | 0.1492 | 0.0963 |
| **C_mf_d256_lr0.001_reg0.001** | **256** | **0.001** | **0.001** | **48** | **0.1522** | **0.0981** |
| C_mf_d128_lr0.003_reg0.001 | 128 | 0.003 | 0.001 | 12 | 0.1376 | 0.0903 |

Observations: Increasing dim from 32 → 128 yields clear gains (NDCG 0.0909 → 0.0929) before **saturating** at 256. `reg = 1e-3` matters more than increasing dim: Stage C (dim 256 + reg 1e-3) reaches **0.0981**, substantially higher than the initial grid's 0.0929. `lr = 3e-3` only performs well when paired with strong regularization; otherwise, it slightly diverges (0.0880). This motivated Stage C, as the initial 2-stage grid only swept regularization at dim 64.

### 5.2 HPO — LightGCN (selected by validation)

| Config | layers | dim | lr | reg | Best epoch | VAL Recall@20 | VAL NDCG@20 |
|---|---|---|---|---|---|---|---|
| A_lgcn_L1_d64 | 1 | 64 | 0.001 | 0.0001 | 54 | 0.1477 | 0.0939 |
| A_lgcn_L2_d64 | 2 | 64 | 0.001 | 0.0001 | 50 | 0.1410 | 0.0916 |
| A_lgcn_L3_d64 | 3 | 64 | 0.001 | 0.0001 | 64 | 0.1400 | 0.0911 |
| A_lgcn_L4_d64 | 4 | 64 | 0.001 | 0.0001 | 79 | 0.1379 | 0.0907 |
| A_lgcn_L2_d32 | 2 | 32 | 0.001 | 0.0001 | 69 | 0.1382 | 0.0909 |
| **A_lgcn_L2_d128** | **2** | **128** | **0.001** | **0.0001** | **100** | **0.1612** | **0.0999** |
| B_lgcn_lr0.001_reg0.0003 | 2 | 64 | 0.001 | 0.0003 | 52 | 0.1428 | 0.0921 |
| B_lgcn_lr0.001_reg0.001 | 2 | 64 | 0.001 | 0.001 | 70 | 0.1417 | 0.0921 |
| B_lgcn_lr0.003_reg0.0001 | 2 | 64 | 0.003 | 0.0001 | 50 | 0.1552 | 0.0975 |
| B_lgcn_lr0.003_reg0.0003 | 2 | 64 | 0.003 | 0.0003 | 52 | 0.1566 | 0.0984 |
| B_lgcn_lr0.003_reg0.001 | 2 | 64 | 0.003 | 0.001 | 76 | 0.1564 | 0.0991 |

Observations:
- At dim 64, **more layers actually degrade performance**: L1 0.0939 > L2 0.0916 > L3 0.0911 > L4 0.0907.