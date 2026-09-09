# Agglomerative Hierarchical Clustering from Scratch — Mall Customer Segmentation

An end-to-end machine learning project implementing Agglomerative Hierarchical Clustering using only NumPy, applied to customer segmentation on the [Mall Customers dataset](https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv) (200 customers).

Built to understand hierarchical clustering's full mathematical machinery; pairwise distance computation, the four major linkage criteria, the Lance–Williams update for Ward linkage, and dendrogram construction from raw merge history, with no reliance on `sklearn.cluster.AgglomerativeClustering` or `scipy.cluster.hierarchy`'s linkage functions until the custom implementation was fully derived and validated.

---

## Project Overview

Unlike a supervised model, hierarchical clustering has no target to fit against and no single global objective. It is a greedy, bottom-up merge process whose entire output is a tree (the dendrogram), from which any flat number of clusters can be cut out after the fact. This project implements that idea from first principles:

1. **Mathematical Derivation** — pairwise Euclidean distance, the four linkage criteria (single, complete, average, Ward), and the Lance–Williams formula that lets Ward's merge cost be updated in closed form without touching the raw points again, worked out in full before any code is written
2. **From-Scratch Implementation** — a `HierarchicalClusteringFromScratch` class (NumPy only) with a vectorized pairwise-distance routine, singleton cluster initialization, closest-pair search, cluster merging, merge-history bookkeeping, and dynamic flat-label extraction at any requested `k`
3. **Unit Testing** — 18 hand-verifiable tests covering distance correctness (a 3-4-5 triangle), each linkage's recovery of an obvious two-blob toy dataset, exact-`k` label extraction, and defensive edge cases (duplicate points, single-observation input, invalid `linkage`/`n_clusters`)
4. **Dendrogram Construction** — the custom merge history converted into SciPy's linkage-matrix format purely as a drawing convenience; the clustering itself is never delegated to SciPy or sklearn

This project walks through the full unsupervised pipeline:

- **Exploratory Data Analysis (EDA)** — feature distributions, boxplot outlier screening, gender balance, pairwise scatter plots, and a correlation matrix confirming the Income/Spending relationship is non-linear (all |r| < 0.3) rather than absent
- **Preprocessing** — manual NumPy standardization ($z = (x-\bar{x})/\sigma$), `CustomerID` excluded as a non-behavioral identifier, `Gender` held out of clustering inputs and reserved for post-hoc profiling, no train/test split (justified explicitly — there is no supervised target to hold out against)
- **Model Implementation** — the `HierarchicalClusteringFromScratch` class described above
- **Linkage Comparison (Experiment 4)** — all four linkage strategies compared on the same dendrogram: single linkage visibly chains, Ward gives the cleanest, most decisive top-level splits
- **Feature-Set Comparison (Experiment 3)** — Income + Spending Score (2D) vs. Income + Spending Score + Age (3D), evaluated across every candidate `k`
- **Cluster-Count Selection (Experiment 5)** — Silhouette, Calinski-Harabasz, and Davies-Bouldin scores swept across k=2–10, cross-checked against the dendrogram's visual cut point
- **Validation (Experiment 6)** — matched comparison against `sklearn.cluster.AgglomerativeClustering`, using label-invariant metrics (Adjusted Rand Index, Normalized Mutual Information) rather than raw label equality, since cluster-label integers are permutation-invariant
- **Cross-Algorithm Benchmarking (Experiment 7)** — K-Means, Gaussian Mixture Models, and DBSCAN run on the same standardized features and compared against the custom hierarchical result
- **Stability & Sensitivity Analysis** — 30-resample bootstrap stability check, plus a full {linkage × feature-set} sensitivity sweep against the chosen baseline
- **Cluster Profiling & Business Insights** — final segments assigned back to the raw dataframe, profiled by size, age, income, spending, and gender composition, then translated into marketing-relevant segment names, only after inspecting the actual statistics, never assumed in advance

Key issues encountered and resolved during the project:

- **Cluster labels are permutation-invariant — element-wise comparison against sklearn would have been meaningless.** Adjusted Rand Index and Normalized Mutual Information (both 1.0000) plus a contingency table are used, instead of checking whether integer label `2` in the custom implementation equals integer label `2` in sklearn's.
- **Adding a feature was tested, not assumed to help.** The full k=2–10 sweep was ran on both the 2D (Income+Spending) and 3D (+Age) feature sets; Age *decreased* Silhouette, Calinski-Harabasz, and Davies-Bouldin scores at nearly every `k`, so it was excluded from the clustering inputs and kept only for post-hoc profiling.
- **The best k by one metric was not the chosen k.** Calinski-Harabasz peaks at k=9 (252.9), not k=5 (244.4), but Silhouette and Davies-Bouldin both peak at k=5, and the dendrogram's clearest visual gap and the EDA scatter plot both independently point to five groups, so k=5 was selected as the configuration with the strongest combined evidence, not the one maximizing a single number.
- **A visually appealing dendrogram cut was checked quantitatively, not trusted on sight.** Flagged a candidate 5-cluster cut from the dendrogram alone; committed to it after the internal validation metrics independently agreed.
- **DBSCAN's disagreement with the other four methods was investigated, not treated as an error.** DBSCAN defines a cluster by density-connectivity rather than distance-to-centroid, so its lower ARI (0.8515) against the hierarchical baseline, and the 23 points it flagged as noise, reflects a genuinely different notion of "cluster," not a bug in either method.

---

## Repository Structure

```
2. Hierarchical_Clustering_mall_dataset/
├── notebook/
│   └── Hierarchical_Clustering_From_Scratch.ipynb
├── README.md
└── requirements.txt
```

> **Note on data:** The dataset is loaded directly from a hosted GitHub CSV mirror inside the notebook, so no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/2. Unsupervised_Learning/2. Hierarchical_Clustering_mall_dataset"
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

See `requirements.txt`. Core libraries used:

| Library | Purpose |
|---|---|
| `numpy` | Pairwise distance computation, standardization, all linkage math, Ward's Lance–Williams update |
| `pandas` | Data loading, structural audit, EDA, cluster profiling |
| `matplotlib` | Visualizations |
| `seaborn` | Distribution plots, correlation heatmap |
| `scipy` | Dendrogram drawing only, rendering the custom merge history, never computing it |
| `scikit-learn` | Reference `AgglomerativeClustering` for validation, `KMeans`/`GaussianMixture`/`DBSCAN` for benchmarking, and evaluation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI) |

---

## Final Model Configuration

| Hyperparameter | Value |
|---|---|
| Linkage | Ward |
| Number of clusters (k) | 5 |
| Feature set | `Annual Income (k$)`, `Spending Score (1-100)` (standardized, z-score) |

Selected via combined evidence: the Silhouette-Score maximum (0.554) and Davies-Bouldin minimum (0.578) both occur at k=5, Calinski-Harabasz at k=5 (244.4) sits close to its k=9 maximum (252.9), and the choice agrees with both the clearest gap in the Ward dendrogram and the five visually distinct blobs seen in the Income vs. Spending scatter plot during EDA, not selected by optimizing any single metric in isolation.

---

## Limitations

- **The naive implementation does not scale past a few hundred customers**: repeatedly scanning all active cluster pairs is worst-case $O(n^3)$, and the full pairwise distance matrix is $O(n^2)$ in memory, fine for 200 customers, but a production version would need a proper nearest-neighbor-chain algorithm to handle a mall's full customer base.
- **The sample is small (200 customers)** and may not represent the mall's entire population, seasonal variation, or non-card-based shoppers.
- **`Spending Score` is a synthetic, mall-assigned metric**, not a direct measure of transaction value, its exact construction methodology is unknown, which limits how literally its business interpretation should be taken.
- **Clustering finds statistical association, not causation**: a "high income, low spending" segment does not by itself explain *why* those customers spend less.
- **Clusters are a snapshot, not a permanent customer identity** — segment membership will drift as behavior changes over time, and this analysis was not repeated across multiple time periods.
- **Age was tested and excluded as a clustering input**: adding it did not improve, and generally worsened, internal validation scores relative to Income + Spending Score alone; it is retained only for descriptive profiling.
- **DBSCAN's parameters (`eps=0.35`, `min_samples=5`) were set by a simple heuristic**, not a full grid search, its comparison result should be read as indicative of a different clustering paradigm, not as a fully tuned DBSCAN baseline.

---

## Results

**Cluster-count selection — internal validation metrics across k=2–10 (Ward, Income + Spending Score):**

| k | Silhouette ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ |
|---|---|---|---|
| 2 | 0.384 | 86.96 | 0.854 |
| 3 | 0.461 | 143.78 | 0.707 |
| 4 | 0.493 | 169.68 | 0.671 |
| **5** | **0.554** | 244.41 | **0.578** |
| 6 | 0.539 | 233.31 | 0.645 |
| 7 | 0.520 | 237.26 | 0.713 |
| 8 | 0.431 | 250.59 | 0.778 |
| 9 | 0.438 | **252.93** | 0.776 |
| 10 | 0.434 | 252.52 | 0.767 |

k=5 wins on two of three metrics outright and sits within 3.5% of the Calinski-Harabasz maximum, which is why it was selected over k=9.

**2D (Income+Spending) vs. 3D (+Age) — same k range:**

| k | Silhouette 2D | Silhouette 3D | Davies-Bouldin 2D | Davies-Bouldin 3D |
|---|---|---|---|---|
| 2 | 0.384 | 0.318 | 0.854 | 1.308 |
| 5 | **0.554** | 0.390 | **0.578** | 0.916 |
| 8 | 0.431 | 0.366 | 0.778 | 0.842 |
| 10 | 0.434 | 0.381 | 0.767 | 0.885 |

The 2D feature set outperforms the 3D set on Silhouette and Davies-Bouldin at every tested k — adding Age dilutes rather than sharpens the segmentation.

**Final cluster profile (k=5, Ward, Income + Spending Score):**

| Cluster | Size | Mean Age | Mean Income (k$) | Mean Spending | % Female | Interpretation |
|---|---|---|---|---|---|---|
| 0 | 21 | 25.3 | 25.1 | 80.0 | 57.1% | Low income, high spending (younger) |
| 1 | 23 | 45.2 | 26.3 | 20.9 | 60.9% | Low income, low spending |
| 2 | 85 | 42.5 | 55.8 | 49.1 | 60.0% | Average income, average spending (largest, "typical") |
| 3 | 39 | 32.7 | 86.5 | 82.1 | 53.8% | High income, high spending (premium/VIP) |
| 4 | 32 | 41.0 | 89.4 | 15.6 | 43.8% | High income, low spending (engagement opportunity) |

**Custom implementation vs. sklearn `AgglomerativeClustering`** (same standardized features, same linkage, same k):

| Metric | Custom | sklearn |
|---|---|---|
| Silhouette | 0.5538 | 0.5538 |
| Calinski-Harabasz | 244.4103 | 244.4103 |
| Davies-Bouldin | 0.5779 | 0.5779 |
| Adjusted Rand Index | 1.0000 | — |
| Normalized Mutual Information | 1.0000 | — |

Perfect agreement — every internal validation metric matches to four decimal places, and the two label sets are identical up to permutation.

**Cross-algorithm benchmark:**

| Algorithm | Clusters found | Silhouette | Davies-Bouldin | ARI vs. custom |
|---|---|---|---|---|
| Custom Hierarchical (Ward) | 5 | 0.5538 | 0.5779 | 1.0000 |
| sklearn AgglomerativeClustering | 5 | 0.5538 | 0.5779 | 1.0000 |
| K-Means | 5 | 0.5547 | 0.5722 | 0.9420 |
| Gaussian Mixture Model | 5 | 0.5537 | 0.5760 | 0.9850 |
| DBSCAN (`eps=0.35`, `min_samples=5`) | 6 (+ 23 noise points) | 0.5577 | 0.5106 | 0.8515 |

K-Means and GMM substantially agree with the hierarchical result; DBSCAN diverges more, consistent with its different (density-based, not centroid-based) definition of a cluster.

**Bootstrap stability** (30 resamples, ARI between the original and each bootstrap-resample clustering, restricted to points common to both):

| Statistic | Value |
|---|---|
| Mean ARI | 0.949 |
| Std | 0.052 |
| Min | 0.828 |
| Max | 1.000 |

**Sensitivity analysis** (ARI against the chosen baseline — Ward, 2D, k=5):

| Linkage | 2D (Income+Spending) | 3D (+Age) |
|---|---|---|
| Single | 0.013 | 0.018 |
| Complete | 0.860 | 0.561 |
| Average | 0.692 | 0.617 |
| Ward | 1.000 (baseline) | 0.581 |

Single linkage is the clear outlier among the four (its dendrogram visibly chains rather than forming compact groups); among the other three, the feature set (2D vs. 3D) moves the result more than the specific linkage choice does.

**Data pipeline:** 200 raw customer records → 0 missing values, 0 duplicated rows, 0 duplicate `CustomerID`s → `CustomerID` excluded (identifier), `Gender` excluded from clustering inputs → standardized `Annual Income (k$)` + `Spending Score (1-100)` → Ward linkage, k=5.

---

## What I learned

1. **The Lance–Williams Update Is What Makes Ward Linkage Practical, Not Just Elegant.**

Recomputing every cluster's within-cluster sum of squares from scratch after each merge would make the algorithm needlessly expensive. Deriving and implementing the closed-form update — the new merged cluster's distance to every remaining cluster expressed purely in terms of the *previous* pairwise distances and cluster sizes, made the connection between the abstract "minimize the increase in within-cluster variance" definition and an actual efficient algorithm concrete in a way the formula alone doesn't.

2. **Chaining Is Not a Textbook Abstraction — It Shows Up Immediately on Real Data.**

Single linkage's Adjusted Rand Index against the Ward baseline was 0.013 — essentially no agreement at all. Watching its dendrogram produce a long staircase of near-identical merge heights, instead of a few tall, decisive splits, turned "single linkage is prone to chaining" from a caveat in a textbook into something directly visible in the notebook's own output.

3. **More Features Is Not Automatically Better for Distance-Based Clustering.**

Adding Age to Income and Spending Score didn't sharpen the five-blob structure, it diluted it. Silhouette and Davies-Bouldin were worse for the 3D feature set at every single tested k. Given the near-zero pairwise correlations found in EDA, in hindsight this makes sense: a weakly-related third dimension mostly adds distance-diffusing noise rather than new separating structure. Running the actual k-sweep on both feature sets, instead of assuming "more context helps," is what surfaced this.

4. **Comparing Two Unsupervised Label Sets Requires a Permutation-Invariant Metric, Not Equality.**

Cluster ID `0` in the custom implementation has no reason to correspond to cluster ID `0` in sklearn's, even a perfect clustering agreement can look like total disagreement under naive label comparison. Using Adjusted Rand Index and Normalized Mutual Information instead (both landing at a clean 1.0000) was the only way to actually confirm the implementations agreed, rather than appearing to disagree due to arbitrary label numbering.

5. **Convergent Agreement Across Independently-Motivated Algorithms Is Stronger Evidence Than Any Single Metric.**

No ground-truth labels exist for this dataset, so no single number can "prove" the segmentation is correct. But K-Means, a Gaussian Mixture Model, and the from-scratch hierarchical clustering, three methods built on different mathematical assumptions about what a cluster is, converged on essentially the same five groups. That convergence, not any one internal validation score, is what makes the five-segment structure credible as a real property of the data rather than an artifact of one algorithm's biases.

6. **Choosing k By Committee Beats Choosing It By the Single Best Metric.**

Calinski-Harabasz's maximum was at k=9, not the k=5 that was ultimately chosen. Silhouette and Davies-Bouldin both preferred k=5 outright, and it also matched the dendrogram's clearest visual gap and the EDA scatter plot's obvious structure. Picking k=5 required weighing multiple, sometimes-disagreeing signals rather than mechanically maximizing whichever metric happened to be computed, and produced a more defensible, more business-interpretable choice than chasing the single highest Calinski-Harabasz score would have.

7. **A Clustering Result Isn't Trustworthy Until It Survives Resampling.**

A clean scatter plot and good internal validation scores describe one specific 200-customer sample. The 30-resample bootstrap check (mean ARI 0.949) is what turns "these clusters look good" into "these clusters are a stable property of the data, not an artifact of exactly which 200 customers happened to be in this dataset", a distinction that matters before recommending a segmentation for real marketing decisions.

---

## Author

**Chinonso Franklin Nwankwo**
[LinkedIn](https://www.linkedin.com/in/chinonso-nwankwo/)