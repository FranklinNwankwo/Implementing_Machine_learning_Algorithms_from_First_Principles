# Principal Component Analysis from Scratch — Wine Dataset Dimensionality Reduction

An end-to-end machine learning project implementing Principal Component Analysis using only NumPy, applied to dimensionality reduction and feature-structure analysis on the [UCI Wine dataset](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/wine_data.csv) (178 samples, 13 chemical measurements, 3 cultivars).

Built to understand PCA's full mathematical machinery; centering, standardization, covariance construction, symmetric eigendecomposition, eigenvalue sorting, variance-based component selection, projection, and reconstruction, with no reliance on `sklearn.decomposition.PCA` until the custom implementation was fully derived, unit-tested, and validated on its own.

---

## Project Overview

Unlike a supervised model, PCA has no target to fit against, it is an unsupervised search for the orthogonal directions of maximum variance in the data, and the wine cultivar label is withheld from it entirely. This project implements that idea from first principles:

1. **Mathematical Derivation** — centering, the covariance matrix $\Sigma = \frac{1}{n-1}X_c^\top X_c$, the eigenvalue problem $\Sigma v = \lambda v$, explained/cumulative variance ratio, projection, and reconstruction, worked out in full before any code is written
2. **From-Scratch Implementation** — a `PCAFromScratch` class (NumPy only) with explicit centering, optional standardization, covariance construction, `numpy.linalg.eigh`-based eigendecomposition, descending eigenvalue sorting, component selection, `transform`, and `inverse_transform`
3. **Unit Testing** — a synthetic 2-feature dataset with a known dominant direction (recovered to within 0° after sign alignment), 13 hand-verifiable property tests (centering, standardization, orthogonality, variance conservation, reconstruction), and 5 edge cases (single component, all components, over-requested components, a constant feature, near-perfectly correlated features)
4. **sklearn Validation** — the custom implementation compared against `sklearn.decomposition.PCA` only after it was independently complete, using sign-ambiguity-aware component comparison rather than raw elementwise subtraction

This project walks through the full unsupervised pipeline:

- **Exploratory Data Analysis (EDA)** — feature distributions, standardized boxplots for outlier screening, a correlation heatmap confirming strong redundancy (e.g. Flavanoids ↔ Total phenols ↔ OD280/OD315), and pairwise scatter plots of chemically related features
- **Preprocessing** — target (`target`, the cultivar label) separated from the feature matrix before anything else touches PCA; 0 missing values and 0 duplicate rows confirmed programmatically, not assumed; mild outliers identified but deliberately retained
- **Model Implementation** — the `PCAFromScratch` class described above
- **Standardization Comparison (Experiments 2–3)** — PCA fit on raw features vs. standardized features on the same data, to make the scale-sensitivity of PCA concrete rather than theoretical
- **Variance Retention (Experiment 4)** — minimum components required to cross the 80% / 90% / 95% / 99% cumulative explained variance thresholds
- **2D & 3D Projection (Experiments 5–6)** — wine samples projected onto PC1–PC2 and PC1–PC2–PC3, colored by cultivar only *after* the unsupervised fit, purely for external validation
- **Loading Analysis (Experiment 7)** — which original chemical variables drive each of the leading three components, read from the full loading pattern rather than any single feature
- **From-Scratch vs. sklearn (Experiment 8)** — explained variance ratio, eigenvalues, component directions (sign-aligned), transformed coordinates, and reconstructions all compared numerically
- **Downstream Clustering (Experiment 9)** — K-Means and Gaussian Mixture Models run on both the full standardized 13D space and the 2D PCA subspace, scored against the withheld cultivar labels
- **Alternative Dimensionality Reduction (Experiment 10)** — Kernel PCA (RBF) and t-SNE as nonlinear benchmarks, and LDA as an explicitly supervised contrast
- **Stability Analysis** — 200-resample bootstrap check on PC1's direction
- **Business Interpretation** — dominant chemical axes and practical applications, written only from the loading table's actual pattern, never assumed in advance

Key issues encountered and resolved during the project:

- **Eigenvector sign ambiguity meant naive comparison against sklearn would have looked broken when it wasn't.** Because $v$ and $-v$ represent the identical principal direction, 8 of the 13 components (PC2, PC4, PC7, PC8, PC9, PC10, PC12, PC13) needed a sign flip before they matched sklearn's chosen orientation. Comparing each component against whichever sign agreed better — not raw elementwise subtraction — is what turned a `1.15e-14` max component difference from "looks like disagreement" into "matches to floating-point precision."
- **Standardization was tested, not assumed necessary.** Fitting PCA on raw (unstandardized) features let a single large-scale variable, Proline, claim 99.8% of "explained variance" on PC1 alone, a unit-of-measurement artifact, not chemical insight. After standardizing, PC1's leading contributor became Flavanoids and variance spread far more evenly across the leading components (36.2% on PC1 instead of 99.8%).
- **A ~2.66e-02 raw eigenvalue gap against sklearn was traced to a ddof convention, not treated as a bug.** The custom standardization step uses `ddof=1`; `sklearn.preprocessing.StandardScaler` uses `ddof=0`. That's a uniform scalar factor across all 13 features, so it shifts every raw eigenvalue by the same ratio but cancels out completely in the explained variance *ratio* (which matched sklearn's to `1.67e-16`), confirming the ratio, not the raw eigenvalue, is the number to trust when comparing two independent PCA implementations.
- **Dimensionality reduction was checked against downstream clustering quality, not assumed to help or hurt it.** K-Means and GMM Silhouette scores nearly doubled when clustering on the 2D PCA subspace instead of the full 13D standardized space (K-Means: 0.285 → 0.561), while agreement with the true cultivar labels (Adjusted Rand Index, Normalized Mutual Information) barely moved, evidence the 11 discarded dimensions were mostly adding noise to the distance calculation, not signal.
- **LDA's cleaner class separation was not mistaken for PCA underperforming.** In the Experiment 10 benchmarking plot, LDA visibly separates the three cultivars more cleanly than PCA — but LDA is handed the cultivar labels directly and optimizes for separability, so its result is a supervised contrast, not a stronger unsupervised projection.

---

## Repository Structure

```
PCA_wine_dataset/
├── notebook/
│   └── PCA_Wine_Dataset.ipynb
├── README.md
└── requirements.txt
```

> **Note on data:** The dataset is loaded directly, inside the notebook, from scikit-learn's own version-controlled GitHub mirror of the UCI Wine dataset, no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/2. Unsupervised_Learning/3. PCA_wine_dataset"
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
| `numpy` | Centering, standardization, covariance construction, `eigh`-based eigendecomposition, projection, reconstruction — the entire `PCAFromScratch` implementation |
| `pandas` | Data loading, structural audit, EDA, loading-matrix and reconstruction-error tables |
| `matplotlib` | Visualizations |
| `seaborn` | Distribution plots, correlation heatmap, pairwise scatter plots |
| `scikit-learn` | Reference `PCA` for validation, `KMeans`/`GaussianMixture`/`KernelPCA`/`TSNE`/`LinearDiscriminantAnalysis` for benchmarking, and evaluation metrics (Silhouette, Adjusted Rand Index, Normalized Mutual Information) |

---

## Final PCA Configuration

| Setting | Value |
|---|---|
| Standardization | Enabled (`ddof=1`) — confirmed necessary; see Results |
| Eigen-solver | `numpy.linalg.eigh` (covariance matrix is symmetric) |
| Components for 2D / 3D visualization | 2 / 3 |
| Components for 80% / 90% / 95% / 99% variance retention | 5 / 8 / 10 / 12 (of 13) |

Standardization was selected over raw-feature PCA by direct comparison, not by default: without it, PC1 is a near-single-feature axis (99.8% of variance, dominated by Proline); with it, variance is spread across the leading components in a way that reflects genuine multi-feature chemical structure rather than measurement units.

---

## Limitations

- **PCA captures linear structure only.** Kernel PCA (RBF kernel) produced a broadly similar 2D picture to linear PCA in this dataset, suggesting the dominant structure here happens to be close to linear, that won't hold for every dataset PCA is applied to.
- **The from-scratch eigendecomposition is a direct `eigh` call on the full $p \times p$ covariance matrix**, fine at $p=13$, but not the randomized/truncated approach a very high-dimensional feature space would need.
- **The sample is modest (178 wines) from three known cultivars in one region**, the bootstrap stability result speaks to this specific dataset, not to generalization across wine populations.
- **Explained variance is not the same thing as predictive usefulness for a specific downstream task.** A low-variance direction PCA discards could still matter for some question other than the one asked here.
- **Component interpretation stops at "phenolic/flavor axis"-level description, not causal chemistry.** PCA identifies directions of variance, not the chemical processes producing that variance.
- **t-SNE's embedding is not variance-preserving** and shouldn't be read on the same axes-have-meaning basis as PCA's components.
- **LDA's cleaner separation uses the cultivar labels directly** — a supervised benchmark, not a stronger version of the unsupervised problem PCA is solving.

---

## Results

**Explained variance by component (standardized PCA, all 13 components):**

| PC | Eigenvalue | Explained Variance Ratio | Cumulative |
|---|---|---|---|
| 1 | 4.706 | 36.20% | 36.20% |
| 2 | 2.497 | 19.21% | 55.41% |
| 3 | 1.446 | 11.12% | 66.53% |
| 4 | 0.919 | 7.07% | 73.60% |
| 5 | 0.853 | 6.56% | 80.16% |
| 6 | 0.642 | 4.94% | 85.10% |
| 7 | 0.551 | 4.24% | 89.34% |
| 8 | 0.348 | 2.68% | 92.02% |
| 9 | 0.289 | 2.22% | 94.24% |
| 10 | 0.251 | 1.93% | 96.17% |
| 11 | 0.226 | 1.74% | 97.91% |
| 12 | 0.169 | 1.30% | 99.20% |
| 13 | 0.103 | 0.80% | 100.00% |

**Variance retention thresholds:**

| Threshold | Components needed | Actual cumulative variance |
|---|---|---|
| 80% | **5** | 80.16% |
| 90% | **8** | 92.02% |
| 95% | **10** | 96.17% |
| 99% | **12** | 99.20% |

**Standardization comparison — top PC1 contributor:**

| | Without standardization | With standardization |
|---|---|---|
| Top PC1 feature | Proline (\|loading\| = 0.9998) | Flavanoids (loading = −0.42) |
| PC1 explained variance | 99.8% | 36.2% |

**Leading component loadings (top 3 by \|loading\|, sign as fitted):**

| Component | Top contributors |
|---|---|
| PC1 | Flavanoids (−0.42), Total phenols (−0.39), OD280/OD315 of diluted wines (−0.38) |
| PC2 | Color intensity (+0.53), Alcohol (+0.48), Proline (+0.36) |
| PC3 | Ash (−0.63), Alcalinity of ash (−0.61), Alcohol (+0.21) |

**Custom implementation vs. sklearn `PCA`** (same standardized features, all 13 components):

| Metric | Value |
|---|---|
| Max \|explained variance ratio difference\| | 1.67e-16 |
| Max \|cumulative EVR difference\| | 1.11e-16 |
| Max \|raw eigenvalue difference\| | 2.66e-02 (ddof convention — see "Key issues," EVR unaffected) |
| Components requiring a sign flip to align | 8 of 13 (PC2, PC4, PC7, PC8, PC9, PC10, PC12, PC13) |
| Max \|component difference\| after sign alignment | 1.15e-14 |
| Min correlation across matched transformed components | 1.000000 |
| Max reconstruction difference (all 13 components) | 5.68e-13 |

Effectively perfect agreement once sign ambiguity and the standardization-ddof convention are accounted for.

**Advanced validation checks (fitted `pca` object):**

| Check | Result |
|---|---|
| Max deviation of $V^\top V$ from identity (orthogonality) | 1.22e-15 |
| Total variance (trace of covariance matrix) | 13.000000 |
| Sum of eigenvalues | 13.000000 |
| Covariance matrix asymmetry | 0.00e+00 |

**Reconstruction error vs. number of components retained:**

| k | MSE | RMSE |
|---|---|---|
| 1 | 4,665.5 | 68.30 |
| 2 | 2,139.7 | 46.26 |
| 3 | 1,962.6 | 44.30 |
| 4 | 1,585.4 | 39.82 |
| 5 | 1,417.0 | 37.64 |
| 6 | 1,347.2 | 36.70 |
| 7 | 1,321.6 | 36.35 |
| 8 | 1,283.3 | 35.82 |
| 9 | 556.5 | 23.59 |
| 10 | 506.4 | 22.50 |
| 11 | 8.29 | 2.88 |
| 12 | 0.176 | 0.419 |
| 13 | ~0 | ~3.85e-14 |

Reconstruction error falls monotonically as more components are retained, reaching numerical zero at k=13 — the mirror image of the cumulative explained-variance curve.

**Downstream clustering — full standardized space vs. PCA subspace (K-Means / GMM, k=3, scored against withheld cultivar labels):**

| Representation | KMeans Silhouette | KMeans ARI | KMeans NMI | GMM Silhouette | GMM ARI | GMM NMI |
|---|---|---|---|---|---|---|
| Full standardized (13D) | 0.285 | 0.897 | 0.876 | 0.285 | 0.897 | 0.876 |
| PCA subspace (2D) | **0.561** | 0.895 | 0.882 | **0.559** | **0.914** | 0.885 |

Cluster *separability* (Silhouette) nearly doubles in the 2D PCA subspace while agreement with the true labels (ARI/NMI) holds steady or improves slightly — the discarded 11 dimensions were mostly noise for this task, not signal.

**Bootstrap stability of PC1's direction** (200 resamples, cosine similarity to the full-dataset PC1):

| Statistic | Value |
|---|---|
| Mean | 0.99377 |
| Std | 0.00530 |
| Min | 0.96364 |

**Data pipeline:** 178 raw wine samples (13 chemical features + 1 withheld cultivar label) → 0 missing values, 0 duplicate rows → target isolated before any PCA fitting → standardized ($z$-score, `ddof=1`) → covariance matrix → `numpy.linalg.eigh` → sorted, sign-ambiguous components. Class distribution: class_0 = 59, class_1 = 71, class_2 = 48.

---

## What I learned

1. **Symmetric Matrices Deserve `eigh`, Not `eig`.**

The covariance matrix is symmetric by construction, and using `numpy.linalg.eigh` instead of the general-purpose `eig` isn't just a performance choice — `eigh` guarantees real-valued eigenpairs and is numerically more stable, where `eig` can return a complex dtype even when the imaginary component is effectively zero. `eigh` also doesn't sort its output descending, which made explicit sorting a step that would have been easy to silently skip.

2. **Explained Variance Ratio Is Scale-Invariant in a Way Raw Eigenvalues Are Not.**

Comparing custom eigenvalues against sklearn's directly showed a `2.66e-02` gap that looked concerning until it traced back to `ddof=1` vs. `ddof=0` in standardization — a uniform scalar factor across all 13 features. The ratio, which divides every eigenvalue by their sum, cancels that factor out entirely (matching to `1.67e-16`), which is exactly why explained variance ratio, not the raw eigenvalue, is the number to trust when comparing two independently written PCA implementations.

3. **A Single Unscaled Feature Can Completely Hijack PCA.**

Fitting on raw features let Proline, values in the hundreds where most other features sit under 20, claim 99.8% of "explained variance" on PC1 alone, purely because of its numerical scale, not because it's the most chemically informative variable. Watching that number appear in the notebook's own output turned "standardize before PCA" from textbook advice into an empirical finding.

4. **Eigenvector Sign Is Genuinely Arbitrary, and Comparisons Must Account For It.**

8 of 13 components needed a sign flip before they'd agree with sklearn's, a direct consequence of $v$ and $-v$ representing the same direction. Naive elementwise subtraction would have made a mathematically correct implementation look broken; aligning each component's sign against whichever orientation matched better before measuring the difference is what turned the comparison into a meaningful `1.15e-14`.

5. **Dimensionality Reduction Can Improve Downstream Clustering, Not Just Compress Storage.**

I expected PCA to trade off some clustering quality for a smaller feature space. Instead, K-Means and GMM Silhouette scores nearly doubled on the 2D PCA subspace versus the full 13D standardized space, while agreement with the true cultivar labels barely moved. The eleven discarded dimensions were apparently adding more noise to the distance calculation than signal.

6. **A Component's Loading Pattern Matters More Than Any Single Feature's Sign.**

It's tempting to name a component after whichever feature has the single largest loading, but that risks a shallow read. PC1's full pattern; Flavanoids, Total phenols, and OD280/OD315 all loading negatively and at comparable magnitude, gave a much clearer sense of "phenolic/flavor chemistry axis" than picking out any one of those three in isolation would have.

7. **Stability Checks Turn "Looks Right" Into "Is Actually Robust."**

A clean scree plot and a tidy sklearn match both describe one specific 178-sample dataset. The 200-resample bootstrap on PC1's direction (mean cosine similarity 0.994, minimum 0.964) is what confirmed the dominant chemical axis is a genuine, stable property of the Wine dataset's structure, not an artifact of exactly which 178 samples happened to be included.

---

## Author

**Chinonso Franklin Nwankwo**
[LinkedIn](https://www.linkedin.com/in/chinonso-nwankwo/)