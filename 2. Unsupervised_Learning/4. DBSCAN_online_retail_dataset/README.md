# DBSCAN from First Principles — Density-Based Customer Segmentation on the UCI Online Retail Dataset

An end-to-end unsupervised machine learning project implementing DBSCAN (Density-Based Spatial Clustering of Applications with Noise) using only NumPy, applied to behavioral customer segmentation on the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (541,909 UK-based transactions, Dec 2010 – Dec 2011, cleaned down to 4,335 customers).

Built to understand DBSCAN's full mechanics; Euclidean distance, epsilon-neighborhood discovery, core/border/noise-point classification, and seed-set-driven cluster expansion, with no reliance on `sklearn.cluster.DBSCAN` until the custom implementation was fully derived, unit-tested on synthetic data, and validated on its own.

---

## Project Overview

Unlike K-Means, DBSCAN has no predetermined cluster count and no assumption that clusters are spherical; it groups points by local density connectivity and explicitly labels points that don't fit any dense neighborhood as noise, rather than forcing every point into a cluster. This project implements that idea from first principles:

1. **Mathematical Derivation** — Euclidean distance, the epsilon-neighborhood $N_\varepsilon(p)$, core/border/noise-point definitions, direct density-reachability, density-reachability (chained, not symmetric), and density-connectivity (symmetric, the relation that actually defines a cluster), worked out in full before any code is written
2. **From-Scratch Implementation** — a `DBSCANFromScratch` class (NumPy only) with a vectorized `_calculate_distance`, `_region_query`, `_is_core_point`, and breadth-first seed-set cluster expansion, using the internal label convention `-2` = unassigned, `-1` = noise, `0, 1, 2, ...` = cluster IDs (`-1` matching sklearn's noise convention, `p` counted in its own neighborhood matching sklearn's core-point convention)
3. **Synthetic Validation** — three purpose-built test cases (separated blobs, interleaved moons, blobs plus scattered noise) plus an exact-duplicate-points case, run against the finished class immediately after it's defined, and 10+ unit tests covering distance correctness, region-query correctness, the core-point boundary, parameter validation, empty/single-point input, and the two degenerate `eps` extremes
4. **sklearn Validation** — the custom implementation compared against `sklearn.cluster.DBSCAN` only after it was independently complete, tested on synthetic data, and applied to real customers, using ARI/NMI (label-permutation-invariant) rather than raw label equality, plus direct noise-mask comparison

This project walks through the full unsupervised pipeline:

- **Structural Data Audit** — shape, dtypes, missingness (`CustomerID` missing on ~25% of rows), duplication, and dataset-specific quirks: cancellation invoices (`C`-prefix), bad-debt adjustment rows (`A`-prefix), non-product stock codes (`POST`, `D`, `M`, `BANK CHARGES`, `PADS`, `DOT`, `CRUK`), and negative-quantity rows that are *not* synonymous with cancellations
- **Data Cleaning** — six ordered, individually-justified rules applied to the raw transactions, each traced back to a specific audit finding rather than applied by default
- **Exploratory Data Analysis (Transaction Level)** — transaction volume over time, country dominance (UK so dominant that `Country` is excluded as a clustering feature), and revenue/quantity distributions
- **Customer-Level Feature Engineering** — transaction rows aggregated to one row per `CustomerID` via RFM (+ behavioral) features: Recency, Frequency (unique invoices), Monetary, AverageOrderValue, TotalQuantity, UniqueProducts
- **Distributions, Skew & Scaling** — skewness measured per feature (1.24 to 41.45), `log1p` applied to five of the six features, correlation checked post-transform, then `StandardScaler` standardization
- **Synthetic Algorithm Validation (Experiment 4)** — the three-plus-one synthetic cases described above, run against the finished `DBSCANFromScratch` class
- **k-Distance Analysis (Experiment 5)** — `sklearn.neighbors.NearestNeighbors` used purely as a k-NN lookup utility (never as part of the clustering algorithm) to find the "elbow" that seeds the `eps` search range
- **Parameter Search (Experiment 6)** — a full grid over `eps` × `min_samples`, run directly through the custom implementation (no sklearn fallback needed at this dataset size), scored by cluster count, noise %, Silhouette, Davies-Bouldin, and Calinski-Harabasz
- **Customer Segmentation (Experiment 7)** — the tuned custom DBSCAN applied to the full standardized, log-transformed 6D feature matrix, visualized via a PCA projection used *strictly* for 2D display, never as clustering input
- **Noise Analysis (Experiment 8)** — the discovered noise population profiled against both clusters on Monetary, Frequency, TotalQuantity, and Recency
- **Cluster Profiling** — the two clusters plus noise translated into named, evidence-based behavioral segments only after inspecting the actual medians
- **sklearn Validation (Experiment 9)** — ARI, NMI, and noise-mask agreement between the custom implementation and `sklearn.cluster.DBSCAN` on identical preprocessed input
- **Cross-Algorithm Benchmarking (Experiment 10)** — K-Means, Agglomerative Clustering, and Gaussian Mixture Models swept over k = 2–6 and compared at a matched k = 3 against DBSCAN's non-noise result
- **Stability & Sensitivity Analysis (Experiment 11)** — `eps`/`min_samples` perturbation around the chosen point, log-transformed vs. raw features at identical parameters, and a cancellation-count sensitivity check
- **Model Assumptions & Limitations, Business Insights, Final Conclusion** — written only from the notebook's own numbers, with all ten initial hypotheses revisited explicitly against what was actually found

Key issues encountered and resolved during the project:

- **Negative `Quantity` is not synonymous with "cancellation."** Every cancellation invoice does have negative `Quantity` (9,288 rows, 100% overlap), but a separate population of 1,336 negative-quantity rows sits on *ordinary* (non-`C`) invoices — these have missing `Description`, `UnitPrice == 0`, and missing `CustomerID`, the signature of manual stock write-offs rather than customer-initiated cancellations. They're excluded for a different reason (no `CustomerID` to attribute them to), not because they resemble cancellations.
- **The 3 `A`-prefix rows are bad-debt write-offs, not purchases or cancellations**, and would have slipped through a naive `InvoiceNo.startswith("C")` filter entirely, so they're excluded explicitly before that filter is even applied.
- **Severe right-skew (11.97–41.45 across five of six features) would have badly distorted what "dense" means for a distance-based algorithm** if left untransformed; `log1p` visibly pulled every affected feature back toward a symmetric, bell-shaped distribution, and a direct raw-vs-log comparison at identical `eps`/`min_samples` later confirmed the two produce materially different segmentations, not a cosmetic difference.
- **`eps` was not chosen by guesswork.** A k-distance plot (`min_samples=10`: ~0.59 at the 70th percentile climbing to ~0.91 by the 90th) located a knee around the 80th–85th percentile, which seeded a structured grid search (`eps` 0.4–1.2, `min_samples` 3–20) rather than a single hand-picked value. The grid confirmed both degenerate extremes predicted by hypothesis #7: `eps=0.4` fragments into 11–17 tiny clusters at 30–75% noise, `eps≥1.0` collapses to one useless mega-cluster at <5% noise.
- **The custom implementation was validated on synthetic ground truth before ever touching customer data.** Case A (3 separated blobs) recovered exactly 3 clusters with an Adjusted Rand Index confirming correct *assignment*, not just correct count. Case B (interleaved moons) separated two non-convex crescents that K-Means provably cannot recover with centroid-based partitioning. Case C (blobs + scattered outliers) correctly flagged the scattered points as noise. An exact-duplicate-points edge case (distance exactly 0) was also tested, since it must not divide by zero or infinite-loop cluster expansion.
- **The noise population turned out to be the *highest*-value group, not the lowest.** Mean `Monetary` for noise (£7,628) is roughly 4× Cluster 0 (£1,844) and 24× Cluster 1 (£314), with mean `TotalQuantity` (4,498 units) dwarfing both clusters. These are plausibly bulk/wholesale buyers whose purchasing volume and frequency simply don't sit inside either "typical customer" density neighborhood, a statistical fact about local density, not a business value judgment, and a direct confirmation (with a twist) of hypothesis #9.
- **Cluster IDs across independent DBSCAN runs are arbitrary permutations, so raw label equality would have been meaningless.** Adjusted Rand Index and Normalized Mutual Information (both label-permutation-invariant) were used instead to compare the custom implementation against sklearn; the noise mask, in contrast, *is* directly comparable since `-1` has a fixed meaning in both.
- **K-Means/Agglomerative/GMM's higher silhouette scores at matched k did not mean they found a better segmentation.** All three score higher than DBSCAN's non-noise result at k=3, expected, since silhouette rewards forcing every point into its nearest centroid — exactly what DBSCAN's noise mechanism refuses to do. Low ARI against DBSCAN (well under 0.3 for all three) confirmed these algorithms weren't recovering the same structure under a different name; none of them has a way to say "482 of these customers don't fit either normal pattern."

---

## Repository Structure

```
4. DBSCAN_online_retail_dataset/
├── notebook/
│   └── DBSCAN_online_retail_dataset.ipynb
├── README.md
└── requirements.txt
```

> **Note on data:** The dataset is loaded directly, inside the notebook, from a GitHub-hosted mirror of the official UCI `Online Retail.xlsx` file, no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/2. Unsupervised_Learning/4. DBSCAN_online_retail_dataset"
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the notebook

```bash
jupyter notebook notebook/
```

---

## Dependencies

See `requirements.txt`. Core libraries used (pinned versions from the notebook's own environment log: pandas 3.0.2, numpy 2.4.4, matplotlib 3.10.8, seaborn 0.13.2, scikit-learn 1.8.0, scipy 1.17.1, Python 3.12.3):

| Library | Purpose |
|---|---|
| `numpy` | Distance computation, epsilon-neighborhood discovery, core/border/noise classification, seed-set cluster expansion — the entire `DBSCANFromScratch` implementation |
| `pandas` | Transaction loading, structural audit, cleaning, RFM aggregation, grid-search and profiling tables |
| `matplotlib` | Visualizations |
| `seaborn` | Distribution plots, correlation heatmap, parameter-grid heatmaps, boxplots |
| `scipy` | Skewness computation (`stats.skew`) during the distributions/scaling analysis |
| `scikit-learn` | `StandardScaler`; `NearestNeighbors` (k-distance diagnostic only, never inside the custom algorithm); reference `DBSCAN`, `KMeans`, `AgglomerativeClustering`, `GaussianMixture`, `PCA` (visualization only) for validation/benchmarking; `make_blobs`/`make_moons` for synthetic tests; Silhouette, Davies-Bouldin, Calinski-Harabasz, Adjusted Rand Index, and Normalized Mutual Information for evaluation |

---

## Final DBSCAN Configuration

| Setting | Value |
|---|---|
| `eps` | 0.7 |
| `min_samples` | 10 |
| Distance metric | Euclidean |
| Feature space | 6D — Recency, Frequency, Monetary, AverageOrderValue, TotalQuantity, UniqueProducts |
| Preprocessing | `log1p` on 5 of 6 features (all but Recency) → `StandardScaler` |
| Clusters found | 2 (+ noise) |
| Noise percentage | 11.1% (482 of 4,335 customers) |
| Silhouette / Davies-Bouldin / Calinski-Harabasz (non-noise points) | 0.289 / 1.241 / 1923 |

`eps=0.7` was chosen because it sits inside the k-distance knee band (~0.65–0.80 for `min_samples=10`); `min_samples=10` follows the common rule-of-thumb of roughly 2× the feature dimensionality (6 features here) and sits mid-grid rather than at either extreme, giving a materially better Silhouette/DBI than the noisier low-`eps` region while still recovering more than one cluster.

---

## Limitations

- **DBSCAN assumes a single global `eps` is adequate**, i.e. roughly uniform density across clusters. If one true segment is much denser than another, a single `eps` can under, or over-cluster one of them, a known limitation distinct from HDBSCAN's variable-density handling.
- **Parameter selection remains partly heuristic.** k-distance analysis and grid search guided `eps`/`min_samples`, but there is no closed-form optimum for either.
- **The naive $O(n^2)$ region-query approach used here would not scale gracefully to millions of points** without spatial indexing (k-d/ball trees).
- **Euclidean distance becomes less discriminating in higher dimensions.** Six features is modest, but this wouldn't hold at, say, 50+ raw product-level features without dimensionality reduction first.
- **The dataset covers roughly one year (Dec 2010–Dec 2011) for a single UK-based retailer** — behavior patterns may not generalize to other retailers, seasons, or time periods.
- **~25% of raw transactions carry no `CustomerID`** and are structurally excluded from ever informing the segmentation.
- **Returns/cancellations were deliberately kept out of the primary feature set**, so revenue attribution around them is intentionally incomplete for this segmentation's purposes.
- **Clusters are descriptive of observed behavior, not causal explanations** of why customers behave that way, and a DBSCAN noise label should never be automatically read as "problem customer."
- **Segments reflect a historical snapshot and will drift**; this is not a live, continuously-updated model, and any resulting marketing action should be validated against real campaign outcomes before being treated as more than a well-supported starting hypothesis.

---

## Results

**Data pipeline:** 541,909 raw transactions → structural audit → cleaning (bad-debt rows, missing `CustomerID`, non-product stock codes, duplicates, cancellation invoices excluded) → 391,316 attributable, non-cancelled purchase-line rows across 4,335 customers → RFM + behavioral aggregation → skew correction (`log1p`) → standardization.

**Cleaning step-by-step effect** (in order applied):

| Step | Rule |
|---|---|
| 1 | Drop the 3 bad-debt adjustment rows (`InvoiceNo` starts with `A`) |
| 2 | Drop rows with missing `CustomerID` (~25% of raw rows) |
| 3 | Drop non-product `StockCode` rows (`POST`, `D`, `M`, `BANK CHARGES`, `PADS`, `DOT`, `CRUK`) |
| 4 | Drop fully duplicated rows |
| 5 | Exclude cancellation invoices (`C`-prefix) from the primary segmentation |
| 6 | Resolve remaining `UnitPrice ≤ 0` rows explicitly (kept — genuine £0 promotional/giveaway line items, all with `Quantity > 0`) |

**Feature skewness (raw RFM features):**

| Feature | Skew |
|---|---|
| Recency | 1.24 |
| UniqueProducts | 6.92 |
| Frequency | 11.97 |
| Monetary | 19.55 |
| TotalQuantity | 20.42 |
| AverageOrderValue | 41.45 |

`log1p` was applied to every feature except Recency (already only mildly skewed, and not a count/monetary quantity with a natural multiplicative spread), visibly pulling the rest back toward symmetric distributions.

**Synthetic validation (before touching real data):**

| Case | Setup | Result |
|---|---|---|
| A | 3 separated Gaussian blobs | 3 clusters recovered, ~0 noise, ARI vs. true labels confirms correct assignment |
| B | 2 interleaved crescents (moons) | 2 clusters recovered — a non-convex shape K-Means cannot recover |
| C | 2 blobs + 20 scattered uniform outliers | 2 clusters found, scattered points correctly flagged as noise |
| Exact-duplicate points | Distance-0 edge case | Handled without divide-by-zero or infinite-loop expansion |

**Parameter grid search behavior:**

| `eps` region | Effect |
|---|---|
| ≤ 0.4 | 11–17 tiny clusters, 30–75% noise — unstable |
| ~0.6–0.8 | 2–3 interpretable clusters, 8–16% noise — the usable middle ground |
| ≥ 1.0 | Collapses toward a single cluster, <5% noise — analytically useless |

**Customer segments discovered** (final `eps=0.7`, `min_samples=10`):

| Segment | n | % | Median Recency | Median Frequency | Median Monetary | Mean Monetary | Mean TotalQuantity |
|---|---|---|---|---|---|---|---|
| Cluster 0 — Core Recurring Customers | 2,522 | 58.2% | 29 days | 4 | — | £1,844 | 1,085 |
| Cluster 1 — One-Time / Lapsed Customers | 1,331 | 30.7% | 125 days | 1 | — | £314 | 189 |
| Noise — High-Volume Outlier Customers | 482 | 11.1% | — | mean 7.8 orders | — | **£7,628** | **4,498** |

**Custom implementation vs. sklearn `DBSCAN`** (identical preprocessed 6D input, `eps=0.7`, `min_samples=10`, Euclidean):

| Metric | Result |
|---|---|
| Clusters found | Match (2, both implementations) |
| Adjusted Rand Index | Near 1.0 |
| Normalized Mutual Information | Near 1.0 |
| Noise-mask agreement | Near 100% of points |

Any residual disagreement lives at cluster boundaries; points at exactly distance `eps`, where a `<=` vs. `<` floating-point convention can tip a borderline point either way, not in the bulk of either cluster.

**Cross-algorithm benchmark** (K-Means / Agglomerative / GMM swept k = 2–6): silhouette peaks at **k=2** for all three algorithms, independently corroborating DBSCAN's 2-real-cluster structure. At a matched k=3, all three alternatives post a *higher* silhouette than DBSCAN's non-noise 2-cluster result, but Adjusted Rand Index against DBSCAN stays well under 0.3 for all three, they are not recovering the same structure under a different name, since none of them has a noise concept and all force the 482 high-volume outliers into whichever cluster is geometrically nearest.

**Stability & sensitivity:**

| Check | Result |
|---|---|
| `eps` ± 0.05–0.1, `min_samples` ± 2 around the chosen point | ARI vs. chosen configuration stays reasonably high, not a knife-edge choice |
| Log-transformed vs. raw features, same `eps`/`min_samples` | Visibly different segmentations — scaling/transformation is not cosmetic for DBSCAN |
| Cancellation counts by segment | Not evenly distributed across groups — flagged for follow-up, deliberately kept out of the primary feature set |

---

## What I learned

1. **Density-Based Clustering Answers a Different Question Than Centroid-Based Clustering.**

DBSCAN doesn't ask "which of k centroids is this point closest to," it asks "is this point in a dense enough neighborhood, possibly reached through a chain of other dense points, to belong to a cluster at all." That's what lets it recover the non-convex moons shape K-Means provably cannot, and, more importantly on real data, what lets it say "482 of these customers don't fit any typical pattern" instead of silently absorbing them into whichever cluster happens to be nearest.

2. **A DBSCAN Noise Label Is a Statement About Local Density, Not About Value.**

I expected noise customers to be low-engagement or low-value, they turned out to be the highest-spending, highest-volume group in the dataset (4–24× the Monetary of either main cluster). They didn't fail the density test because they're bad customers, they failed it because unusually large, unusually frequent purchasing puts them in a sparser region of the feature space than either "typical" customer population.

3. **Cluster-ID Comparisons Need Permutation-Invariant Metrics.**

Two independent DBSCAN runs can produce identical clusterings with completely different integer labels attached. Comparing custom-vs-sklearn output with raw label equality would have looked like disagreement even when the assignments matched perfectly; Adjusted Rand Index and Normalized Mutual Information solved this cleanly, while the noise mask (`-1`) could be compared directly since it has a fixed meaning in both implementations.

4. **Skew Isn't Just a Distributional Curiosity for a Distance-Based Algorithm, It Directly Distorts What "Dense" Means.**

With skewness up to 41.45 on AverageOrderValue, a handful of extreme customers would have dominated every Euclidean distance calculation before scaling ever got a chance to matter. Running the *same* `eps`/`min_samples` on raw vs. log-transformed features and getting a materially different segmentation turned "log-transform skewed features" from textbook advice into a directly observed effect.

5. **A Higher Silhouette Score Doesn't Automatically Mean a Better Segmentation.**

K-Means, Agglomerative, and GMM all posted higher silhouette scores than DBSCAN's non-noise result at a matched k, which makes sense mechanically (silhouette rewards forcing every point into its nearest centroid) but would be the wrong metric to declare them the "better" clustering here. Their low ARI against DBSCAN, and the fact that none of them has any way to flag the 482-customer outlier group as behaviorally distinct, is the more important comparison for the actual business question.

6. **The k-Distance Elbow Is a Starting Hypothesis, Not a Guaranteed Optimum.**

The k-distance plot narrowed the search to a plausible `eps` band, but it was the structured grid search across that band, and the region outside it, scored against real clustering-quality metrics, that actually justified the final `eps=0.7`, `min_samples=10` choice, and that also revealed the two degenerate failure modes (over-fragmentation at low `eps`, one useless mega-cluster at high `eps`) the elbow alone wouldn't have shown.

7. **Synthetic Validation Before Real Data Catches Bugs a Real Dataset Would Hide.**

The exact-duplicate-points edge case (distance exactly 0) is easy to miss when writing a region query, and easy to never notice as broken on messy real-world data where duplicates are rare and errors might just look like "slightly odd clustering." Testing it explicitly on synthetic data, before the implementation ever touched customer data, is what turned "probably correct" into a verified guarantee.

---

## Author

**Chinonso Franklin Nwankwo**
[LinkedIn](https://www.linkedin.com/in/chinonso-nwankwo/)