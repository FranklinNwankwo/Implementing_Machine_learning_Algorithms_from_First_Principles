# K-Means Clustering from Scratch — Iris Species Discovery

An end-to-end machine learning project implementing K-Means clustering using only NumPy, applied to unsupervised species discovery on the [Iris Flower Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset) (150 observations, 4 numerical features, 3 known species).

Built to understand Lloyd's algorithm's full mechanics; centroid initialization, vectorised distance computation, cluster assignment, centroid recomputation, empty-cluster recovery, and convergence detection, with no reliance on `sklearn.cluster.KMeans` for the primary clustering result. Scikit-learn is used only afterward, as an independent benchmark and to run alternative clustering algorithms for comparison.

---

## Project Overview

Given flower measurements *without* their species labels, can K-Means discover meaningful groups based solely on four numerical features? This project implements that idea from first principles:

1. **Mathematical Derivation** — the within-cluster sum of squared distances objective $J$, worked out before any code is written
2. **From-Scratch Implementation** — a `KMeansFromScratch` class (NumPy only) mirroring sklearn's estimator API (`fit`/`predict`), with random-observation centroid initialization, a single vectorised broadcast distance calculation, `argmin`-based cluster assignment, mean-based centroid updates, explicit empty-cluster recovery, and inertia tracking
3. **Unsupervised K Selection** — the elbow method and silhouette analysis across $K = 2, \dots, 10$, evaluated without touching the species labels
4. **Rigorous, Permutation-Aware Evaluation** — because cluster IDs are arbitrary, comparison against ground truth requires solving the label-permutation problem via the Hungarian algorithm before reporting a "mapped accuracy," alongside permutation-invariant metrics (ARI, NMI).

This project walks through the full unsupervised learning pipeline:

- **Exploratory Data Analysis (EDA)** — feature distributions by species, boxplots, pairwise relationships, and a correlation heatmap, with species used strictly for visual interpretation, never as a model input
- **Feature Scaling** — a manual z-score standardisation function, applied to test whether removing the scale disparity between petal and sepal measurements changes clustering quality
- **Model Implementation** — a `KMeansFromScratch` class (vectorised Euclidean distance, empty-cluster re-seeding, convergence via centroid-movement tolerance) built from scratch with NumPy
- **K Selection** — elbow (inertia) and silhouette diagnostics across candidate $K$, computed only on standardised features
- **Initialization Stability Analysis** — the final model configuration re-fit across 30 independent random seeds to quantify how much the result depends on initialization luck versus genuine data structure
- **Validation** — matched comparison against `sklearn.cluster.KMeans` (`init="random"`, `n_init=1`, same seed), isolating the comparison to whether the two Lloyd's-algorithm implementations agree
- **Comparative Modeling** — benchmarked against Agglomerative (Ward) Clustering, a Gaussian Mixture Model, and DBSCAN (with an `eps` sweep), each encoding different assumptions about cluster shape or density
- **External Evaluation** — Adjusted Rand Index, Normalized Mutual Information, and Hungarian-algorithm-mapped cluster accuracy against the withheld species labels, plus a contingency table
- **Diagnostics** — convergence trace inspection, PCA-projected cluster visualisation, and an explicit statement of K-Means's modeling assumptions checked against the evidence gathered

Key issues encountered and resolved during the project:

- **Inertia is not comparable across unscaled vs. scaled runs** — unscaled inertia is in squared centimetres, scaled inertia is in squared standard-deviation units, so a lower raw number on one side does not mean tighter clusters; silhouette score and ARI (both unit-free) were used instead to judge whether standardisation actually helped

- **Silhouette evidence and biological prior knowledge disagreed on $K$** — because *versicolor* and *virginica* overlap substantially, the silhouette curve often favoured $K=2$ over $K=3$; rather than picking whichever $K$ matched the three known species and calling it "unsupervised evidence," the notebook reports this tension honestly and keeps $K=3$ for external-evaluation purposes while stating plainly that silhouette alone does not uniquely mandate it

- **Comparing cluster labels directly is invalid** — cluster IDs are arbitrary (the custom model's "cluster 0" need not correspond to sklearn's "cluster 0"), so `compute_mapped_accuracy` solves the optimal cluster-to-species assignment via `scipy.optimize.linear_sum_assignment` (the Hungarian algorithm) before any accuracy-style number is reported

- **A matched sklearn seed does not guarantee matched initial centroids** — passing the same `random_state` to both implementations isolates whether the *algorithms* agree, not whether they draw identical starting points, since sklearn's internal sampling procedure differs from the custom implementation's; the resulting cluster-size divergence between the two runs is expected, not a bug

- **DBSCAN needed a real `eps` sweep, not a single guess** — a sweep from `eps=0.3` to `1.2` was run and scored by silhouette (on non-noise points) before selecting `eps=1.1`, which collapses *versicolor* and *virginica* into a single density-connected region rather than resolving three clusters, a genuinely different failure mode from K-Means's own struggle at that same boundary

---

## Repository Structure

```
1. K-Means_Iris_flower/
├── data/
│   └── .gitkeep
├── notebook/
│   └── KMeans_Clustering.ipynb
├── README.md
└── requirements.txt
```

> **Note on data:** The dataset is not committed to this repo. It is loaded directly via `sklearn.datasets.load_iris()` inside the notebook, so no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/2. Unsupervised_Learning/1. K-Means_Iris_flower"
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
| `numpy` | Centroid initialisation, vectorised distance computation, cluster assignment, centroid updates, inertia calculation |
| `pandas` | Data manipulation, structural audit, EDA, results tables |
| `matplotlib` | Visualisations |
| `seaborn` | Distribution plots, boxplots, pairplots, correlation heatmap |
| `scikit-learn` | `load_iris`, `KMeans` (benchmark), `AgglomerativeClustering`, `DBSCAN`, `GaussianMixture`, `PCA`, silhouette/ARI/NMI metrics |
| `scipy` | `linear_sum_assignment` (Hungarian algorithm) for permutation-aware mapped accuracy |

---

## Final Model Configuration

| Hyperparameter | Value |
|---|---|
| `n_clusters` (K) | 3 |
| `init` | `random` (K distinct observations drawn from the dataset) |
| `max_iter` | 300 |
| `tol` | 1e-4 |
| Feature representation | Z-score standardised (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) |
| `random_state` | 42 |

$K=3$ was selected to align with the three known species for external evaluation; the unsupervised elbow/silhouette evidence across $K=2,\dots,10$ did not unambiguously prefer $K=3$ over $K=2$ (see Results).

---

## Limitations

- **$K=3$ is a biologically-informed choice, not an unambiguous unsupervised one**: the silhouette curve often scores $K=2$ as well as or better than $K=3$, because *versicolor* and *virginica* are not cleanly separable by geometry alone.
- **Standardisation did not clearly improve external metrics on this dataset**: the naturally larger-magnitude petal measurements are also the most species-informative ones, so scaling away that advantage traded ARI (0.716 → 0.645) for a small silhouette drop as well; standardisation is kept as the more defensible general-purpose default despite this.
- **The mean is not a robust statistic**: K-Means assumes outliers are not overwhelmingly distorting the cluster centroid, an assumption untested here since Iris contains essentially no outliers.
- **Convergence is only to a local optimum**: different random initialisations can converge to different final cluster configurations, quantified in the 30-seed stability experiment.
- **The permutation problem constrains all label-based comparison**: cluster IDs are arbitrary, so any accuracy-style number requires solving an optimal mapping first (Hungarian algorithm); ARI and NMI remain the more fundamentally sound metrics.
- **Ground-truth labels are a luxury**: they exist here purely because Iris is a labelled benchmark; genuine unsupervised problems in practice typically lack any external validation signal.
- **Iris is small and unusually clean**: 150 rows, no missing values, no serious outliers, and perfectly balanced classes. Conclusions about robustness here should not be over-generalised to noisier, larger, or imbalanced production data.

---

## Results

**Baseline: unscaled vs. scaled K-Means ($K=3$):**

| Metric | Unscaled | Scaled |
|---|---|---|
| Inertia | 78.8557 | 140.9015 |
| Silhouette Score | 0.5512 | 0.4565 |
| ARI vs. species | 0.7163 | 0.6451 |
| Iterations to converge | 11 | 5 |

Inertia is not comparable across the two rows (different units); silhouette and ARI, which are unit-free, show the unscaled run scoring *higher* on both — because the larger-magnitude petal measurements happen to also be the most species-informative features, standardising away that scale advantage did not produce a clear external-metric improvement here.

**K selection ($K = 2$ through $10$, standardised features):**

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 222.3617 | 0.5818 |
| **3** | **140.9015** | **0.4565** |
| 4 | 114.5568 | 0.4151 |
| 5 | 104.7447 | 0.3965 |
| 6 | 96.9885 | 0.3744 |
| 7 | 87.9702 | 0.3793 |
| 8 | 83.4260 | 0.3740 |
| 9 | 64.6906 | 0.2967 |
| 10 | 51.2139 | 0.3207 |

The elbow curve decreases smoothly with no single obvious "elbow." $K=2$ scores *higher* on silhouette than $K=3$, a direct consequence of the *versicolor*/*virginica* overlap, so $K=3$ was retained specifically to align with the known species for external evaluation, not because unsupervised evidence alone demanded it.

**Final model ($K=3$, standardised features):**

- Converged in **5 iterations**, final inertia **140.9015**
- Cluster sizes: **46 / 49 / 55**
- Cluster → species mapping (Hungarian algorithm): cluster 0 → versicolor, cluster 1 → setosa, cluster 2 → virginica

| External Metric | Value |
|---|---|
| Silhouette Score | 0.4565 |
| Adjusted Rand Index | 0.6451 |
| Normalized Mutual Information | 0.6613 |
| Mapped cluster accuracy | 0.8533 |

**Contingency table** (cluster assignment vs. true species):

| Cluster | setosa | versicolor | virginica |
|---|---|---|---|
| 0 | 1 | 37 | 8 |
| 1 | 49 | 0 | 0 |
| 2 | 0 | 13 | 42 |

*Setosa* is recovered almost perfectly (49/50 in a single cluster). Nearly all misassignments are cross-contamination between *versicolor* and *virginica*.

**Initialization stability (30 independent seeds, $K=3$, standardised):**

| Metric | Mean ± Std | Min | Max |
|---|---|---|---|
| Inertia | 145.65 ± 16.15 | 139.82 | 197.47 |
| Silhouette | 0.4618 ± 0.0086 | 0.4565 | 0.4951 |
| ARI | 0.6053 ± 0.0588 | 0.4328 | 0.6451 |
| Mapped Accuracy | 0.8107 ± 0.0800 | 0.5800 | 0.8533 |

Most seeds cluster tightly near the best-known solution, but the minimum-inertia outlier (197.47 vs. a median of ~140.5) and the accuracy range (0.58–0.85) confirm that a minority of initialisations converge to a genuinely worse local optimum, the concrete motivation for smarter initialisation schemes like K-Means++.

**Custom `KMeansFromScratch` vs. `sklearn.cluster.KMeans`** (matched `init="random"`, `n_init=1`, same seed):

| Metric | Custom KMeansFromScratch | sklearn KMeans |
|---|---|---|
| Inertia | 140.9015 | 140.0328 |
| Iterations | 5 | 6 |
| Silhouette Score | 0.4565 | 0.4630 |
| Adjusted Rand Index | 0.6451 | 0.5923 |
| Normalized Mutual Info | 0.6613 | 0.6427 |
| Mapped Accuracy | 0.8533 | 0.8133 |
| Cluster sizes | [46, 49, 55] | [56, 50, 44] |

Absolute inertia difference: **0.87** (out of ~140). Cluster *sizes* diverge more visibly than the quality metrics — expected, since a matched `random_state` does not guarantee both implementations draw the same initial centroids from a different internal sampling procedure.

**Consolidated comparison across algorithms** (standardised features):

| Algorithm | K / Parameters | Clusters Found | Silhouette | ARI | NMI |
|---|---|---|---|---|---|
| Custom K-Means (from scratch) | K=3 | 3 | 0.4565 | 0.6451 | 0.6613 |
| sklearn KMeans | K=3, init=random | 3 | 0.4630 | 0.5923 | 0.6427 |
| Agglomerative Clustering | K=3, Ward linkage | 3 | 0.4467 | 0.6153 | 0.6755 |
| Gaussian Mixture Model | 3 components | 3 | 0.4751 | 0.5165 | 0.6571 |
| DBSCAN | eps=1.10, min_samples=5 | 2 (+2 noise) | 0.5518 | 0.5656 | 0.7326 |

Agglomerative (Ward) — which shares K-Means's compact, similarly-sized-cluster preference tracks K-Means closely. DBSCAN, the only density-based method, collapses to **2** clusters rather than 3: it cannot separate *versicolor* from *virginica* at all, the same boundary where K-Means itself is weakest, which is independent evidence that the difficulty is a property of the data rather than a K-Means-specific artefact.

**Data pipeline:** 150 observations loaded via `sklearn.datasets.load_iris()`, no missing values, 1 duplicate row retained (a known duplicate in the classic Fisher dataset between two `virginica` observations), 3 perfectly balanced classes (50 each).

---

## What I learned:

1. **Cluster IDs Are Arbitrary Labels, and Every Downstream Comparison Must Respect That.**

Unlike supervised classification, where predicted label `1` always means the same thing as ground-truth label `1`, K-Means's "cluster 0" carries no inherent meaning. Implementing `compute_mapped_accuracy` from scratch, solving the optimal cluster-to-species assignment via the Hungarian algorithm before computing anything resembling "accuracy," made concrete why ARI and NMI (which are permutation-invariant by construction) are the more fundamentally sound metrics, and why a naive label-matching approach would have been silently wrong.

2. **Internal and External Metrics Can Disagree, and That Disagreement Is Itself a Finding.**

The silhouette curve favoured $K=2$ over $K=3$, while the external ARI/NMI evidence (once species labels were consulted) supported $K=3$ as the biologically meaningful choice. Reporting this tension honestly, rather than picking whichever $K$ matched the known answer and presenting it as the unsupervised result, was a more defensible use of the evaluation framework than collapsing internal and external evidence into a single number.

3. **Inertia Only Means Something Within a Fixed Feature Scale.**

Comparing 78.86 (unscaled) against 140.90 (scaled) directly would have implied scaling made clustering four times worse, when the two numbers are in incompatible units (squared centimetres vs. squared standard-deviation units). Reaching for silhouette and ARI, both unit-free, instead of inertia was necessary to make a fair scaled-vs-unscaled judgment at all.

4. **A Matched Random Seed Does Not Mean Matched Behaviour Across Implementations.**

Passing `random_state=42` to both the custom implementation and `sklearn.cluster.KMeans` does not guarantee identical initial centroids, since the two draw from different internal sampling procedures even under `init="random"`. The resulting cluster-size divergence (46/49/55 vs. 56/50/44) despite near-identical inertia and silhouette was the concrete demonstration that "same seed" isolates algorithmic correctness, not bit-identical output.

5. **Convergence to a Local Optimum Is Not a Hypothetical Caveat.**

Thirty independent seeds mostly converged to (nearly) the same solution, but one seed landed at an inertia of 197.47 against a median of ~140.5, a genuinely worse local optimum. Seeing that outlier directly in the stability boxplots, rather than just reading about K-Means's local-optimum limitation in the abstract, was what made the case for smarter initialisation (K-Means++) tangible.

6. **Comparing Against Algorithms With Different Assumptions Validates (or Challenges) a Result More Than Comparing Against the Same Algorithm Twice.**

Agglomerative Clustering, which shares K-Means's compact-cluster geometry, agreed closely. DBSCAN, a density-based method with no such assumption, could not separate *versicolor* from *virginica* at all and collapsed them into one cluster. Two independent lines of evidence converging on the same weak spot is stronger support for "this is a property of the data" than any single algorithm's result could provide alone.

7. **Empty-Cluster Handling Is a Real Edge Case, Not a Theoretical Nicety.**

Writing `_handle_empty_clusters` to re-seed any centroid that receives zero assigned points, using the currently worst-fit point, forced confronting a failure mode (unlucky initialisation stranding a centroid) that is easy to skip when only ever calling `sklearn.cluster.KMeans`, but that any correct from-scratch implementation of Lloyd's algorithm must handle explicitly to avoid a division-by-zero crash.

---

## Author

**Chinonso Franklin Nwankwo**
[LinkedIn](https://www.linkedin.com/in/chinonso-nwankwo/)