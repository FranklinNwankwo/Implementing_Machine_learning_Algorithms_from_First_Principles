# Multilayer Perceptron from Scratch — MNIST Handwritten Digit Classification

An end-to-end machine learning project implementing a Multilayer Perceptron using only NumPy, applied to 10-class handwritten digit classification on the [MNIST dataset](https://www.openml.org/search?type=data&id=554) (70,000 28×28 grayscale images).

Built to understand the MLP's full mathematical machinery, forward propagation, softmax, categorical cross-entropy, backpropagation via matrix calculus, He initialization, with no reliance on TensorFlow, Keras, PyTorch, or JAX, and no reliance on `sklearn.neural_network` until the custom implementation was fully derived and numerically validated.

---

## Project Overview

Unlike a linear model, an MLP learns a *nonlinear* mapping from raw pixels to class probabilities by composing affine transformations with nonlinear activations across hidden layers. This project implements that idea from first principles:

1. **Mathematical Derivation** — forward propagation ($Z_l = A_{l-1}W_l + b_l$), ReLU and its derivative, a numerically stable softmax, categorical cross-entropy, and the full backpropagation chain (including the $dZ_{\text{out}} = A_{\text{out}} - Y$ softmax+cross-entropy shortcut), worked out in full before any code is written
2. **From-Scratch Implementation** — an `MLPClassifierFromScratch` class (NumPy only) written generically over an arbitrary number of layers, with He-initialized parameters, vectorized forward/backward passes, mini-batch shuffling, and a training loop tracking train/validation loss and accuracy every epoch
3. **Numerical Gradient Verification** — finite-difference checking of every analytical gradient against a tiny synthetic network, confirmed to a maximum relative error of `7.39e-09` across every parameter matrix, as proof of backpropagation correctness independent of any accuracy number
4. **Baseline Architecture** — `784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)` (109,386 trainable parameters), trained with plain mini-batch gradient descent (no momentum, no adaptive learning rate, no explicit regularization) so the baseline reflects the architecture's *unassisted* learning dynamics

This project walks through the full supervised learning pipeline:

- **Exploratory Data Analysis (EDA)** — class balance (0.801 min/max ratio), sample digit grids per class, pixel-intensity distribution (80.8% of pixels are exactly zero background), and pixel-wise average image per digit class
- **Preprocessing** — `[0, 255] → [0, 1]` normalization, one-hot label encoding, and a stratified 90/10 train/validation split carved out of the official 60,000-image training set; the official 10,000-image test set is held out untouched until final evaluation
- **Model Implementation** — an `MLPClassifierFromScratch` class (He initialization, vectorized forward/backward, mini-batch gradient descent) built from scratch with NumPy
- **Gradient Checking** — analytical vs. numerical gradients compared on tiny networks at 1, 2, and 3 hidden layers deep
- **Hyperparameter Sensitivity Analysis** — controlled, one-factor-at-a-time sweeps over learning rate, hidden architecture, batch size, and epoch count
- **Ablations** — network depth, full-batch vs. mini-batch gradient descent, ReLU vs. sigmoid hidden activations
- **Validation** — matched comparison against `sklearn.neural_network.MLPClassifier` with momentum explicitly disabled on both sides, including a full prediction-agreement check
- **Comparative Modeling** — benchmarked against Logistic Regression, KNN, SVM (RBF kernel), Random Forest, and Gradient Boosting
- **Diagnostics** — training/validation loss and accuracy curves, confusion matrix, per-class precision/recall/F1, most-confused digit pairs, and sample misclassified images with prediction confidence

Key issues encountered and resolved during the project:

- **The forward/backward pass was initially hardcoded to exactly two hidden layers** — an early version of `MLPClassifierFromScratch.forward()`/`backward()` explicitly referenced `W1, W2, W3` and `Z1, Z2, Z3`, which silently broke the moment the architecture sweep tried a single-hidden-layer network (`784 → 64 → 10`) or a deeper one (`784 → 256 → 128 → 10`), raising a `KeyError: 'W3'`. The class was rewritten to loop generically over `self.layer_sizes`, and gradient checking was re-run at 1, 2, and 3 hidden-layer depths (relative error on the order of `1e-8`–`1e-9` at every depth) to confirm the generic version was still mathematically correct, not just crash-free

- **The gradient check's own finite-difference method produced a spurious false alarm** — an early run of Section 10's check (fixed seed, biases left at their exact He-initialized value of zero) reported a relative error of `0.163` isolated entirely to `b2`, while every other parameter matrix checked out at `~1e-9`. The root cause was traced to one synthetic example whose entire hidden-1 activation happened to be zero, putting its hidden-2 pre-activation exactly on the ReLU kink (`Z=0`) for every unit simultaneously; a bias perturbation shifts that example's pre-activation by the same amount on every unit at once, which can flip the kink and inflate the finite-difference estimate without the analytical gradient being wrong. Nudging biases slightly off zero before checking (and re-running) resolved it: the confirmed result is `7.39e-09` max relative error, uniformly distributed across all six parameter matrices

- **The sklearn MLP comparison was not actually apples-to-apples** — `sklearn.neural_network.MLPClassifier(solver="sgd", ...)` defaults to `momentum=0.9` with Nesterov acceleration, a materially different (and faster-converging) optimizer than the custom implementation's plain, un-accelerated mini-batch gradient descent. With momentum left at its default, sklearn's MLP beat the custom MLP by 1.5 points (0.9791 vs. 0.9641). Setting `momentum=0.0, nesterovs_momentum=False` to match the custom implementation's optimizer collapsed that gap to 0.18 points (0.9623 vs. 0.9641) — the custom MLP essentially ties, and on this run even edges out, a matched-optimizer sklearn model
- **SVM, KNN, and Gradient Boosting do not scale cleanly to 54,000 training examples** — rather than silently dropping them from the benchmark or training them on the full set at prohibitive cost (single-digit minutes turning into tens of minutes), they are trained on a fixed, stratified 10,000-example subset, still evaluated on the full 10,000-example official test set, with the caveat documented directly next to the comparison table rather than left implicit

---

## Repository Structure

```
10. MLP_MNIST/
├── notebook/
│   └── MLP_MNIST.ipynb
├── README.md
└── requirements.txt
```

> **Note on data:** The dataset is not committed to this repo. It is loaded directly from OpenML via `sklearn.datasets.fetch_openml("mnist_784")` inside the notebook (cached locally by scikit-learn after the first download), so no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/1. Supervised_Learning/10. MLP_MNIST"
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
| `numpy` | Forward/backward propagation, He initialization, softmax, cross-entropy, mini-batch gradient descent — the entire custom MLP |
| `pandas` | Structural audit, EDA, experiment result tables |
| `matplotlib` | Visualizations |
| `seaborn` | Confusion matrix heatmap, class-distribution and pixel-distribution plots |
| `scikit-learn` | `fetch_openml` MNIST loader, `train_test_split`, evaluation metrics, `MLPClassifier` validation model, classical-algorithm benchmark models |

---

## Final Model Configuration

| Hyperparameter | Value |
|---|---|
| Architecture | `784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)` |
| Weight initialization | He initialization ($W \sim \mathcal{N}(0,\, 2/n_{\text{in}})$), zero-initialized biases |
| Learning rate | 0.01 |
| Batch size | 64 |
| Epochs | 20 |
| Optimizer | Plain mini-batch gradient descent (no momentum, no adaptive learning rate) |
| Regularization | None in the baseline (explored separately in the ablations) |

Fixed as the project's specified baseline configuration rather than tuned for maximum accuracy — the learning-rate/architecture/batch-size/epoch sweep in the notebook's hyperparameter experiments section found this to be a reasonable, well-behaved setting, but not the single best one found (`lr=0.05` reached 0.9698 validation accuracy at the same epoch budget vs. 0.9413 for the `lr=0.01` baseline; see Limitations).

---

## Limitations

- **No built-in translation invariance**: a dense MLP treats each of the 784 pixel positions as an independent feature; a stroke shifted by a few pixels is a completely different input vector to every hidden unit. A convolutional architecture would be the natural next step for robustness to spatial shifts.
- **The baseline learning rate is not the best one found**: the hyperparameter sweep (Section 15) shows `lr=0.05` reaching 0.9698 validation accuracy versus 0.9413 for the `lr=0.01` baseline at the same 8-epoch sweep budget — the baseline was kept at the project's originally specified value rather than re-tuned to the sweep's own findings.
- **No momentum or adaptive learning rate**: the baseline uses plain fixed-learning-rate mini-batch gradient descent by design. The full-batch-vs-mini-batch ablation makes the cost of this concrete: full-batch gradient descent, given the same 8-epoch sweep budget as everything else, gets only 8 total parameter updates and lands at 10.8% validation accuracy — essentially chance level, not merely "slower convergence."
- **No explicit regularization in the baseline model**: the train/validation accuracy gap widens from 1.09 points at 20 epochs to 1.50 points at 30 epochs (Section 15's epoch sweep) — a mild, expected overfitting trend for an unregularized two-hidden-layer MLP, not yet severe, but the direction to watch if training were extended further.
- **Classical-benchmark training-set sizes are not uniform**: SVM, KNN, and Gradient Boosting are trained on a 10,000-example stratified subset for computational tractability (Gradient Boosting alone took ~1,082 seconds even at that reduced size), while Logistic Regression, Random Forest, and both MLPs are trained on the full 54,000-example training set — their relative ranking in the results table should be read with that difference in mind.
- **Naive finite-difference gradient checking has its own edge cases**: as documented above, checking a bias parameter can spuriously inflate the finite-difference estimate when a training example's activation happens to sit exactly on a ReLU's non-differentiable point (`Z=0`) for every unit in a layer simultaneously. This doesn't affect the check's validity at normal parameter values (confirmed at `7.39e-09` once biases are nudged off exact zero before checking), but it's worth knowing about before treating any single isolated spike as proof of a bug.

---

## Results

**Test set — full model comparison (sorted by test accuracy):**

| Rank | Model | Accuracy | Precision | Recall | F1 | Train Time |
|---|---|---|---|---|---|---|
| 1 | Random Forest (300 trees) | 0.9691 | 0.9690 | 0.9688 | 0.9689 | 58.3 s |
| 2 | **Custom MLP (from scratch)** | **0.9641** | **0.9640** | **0.9637** | **0.9637** | **56.6 s** |
| 3 | SVM — RBF kernel (10k-subset) | 0.9634 | 0.9633 | 0.9630 | 0.9631 | 21.0 s |
| 4 | sklearn MLPClassifier | 0.9623 | 0.9622 | 0.9618 | 0.9620 | 72.3 s |
| 5 | K-Nearest Neighbors (10k-subset) | 0.9464 | 0.9482 | 0.9458 | 0.9462 | 0.0 s* |
| 6 | Gradient Boosting (10k-subset) | 0.9321 | 0.9316 | 0.9314 | 0.9314 | 1,082.1 s |
| 7 | Logistic Regression | 0.9243 | 0.9234 | 0.9232 | 0.9232 | 36.0 s |

\* KNN is a lazy learner — `.fit()` only stores the training data, so its near-zero training time defers essentially all computational cost to (unmeasured) prediction time. Don't read it as the cheapest model overall.

The custom implementation places 2nd of 7 models on test accuracy, ahead of a matched-optimizer sklearn `MLPClassifier`, ahead of SVM despite SVM training on a smaller subset, and behind only Random Forest, whose strong showing on raw pixels was a genuinely unanticipated result (see Section 20 of the notebook).

**Custom vs. sklearn MLP, matched optimizer (`momentum=0`, `nesterovs_momentum=False` on both sides):**

| Metric | Custom MLP | sklearn MLP | Δ |
|---|---|---|---|
| Accuracy | 0.9641 | 0.9623 | +0.0018 |
| Precision | 0.9640 | 0.9622 | +0.0018 |
| Recall | 0.9637 | 0.9618 | +0.0019 |
| F1 Score | 0.9637 | 0.9620 | +0.0017 |
| Train Time | 56.6 s | 72.3 s | −15.7 s |

**Prediction agreement: 97.66%** of test samples — both implementations reach the same class assignment for the large majority of the 10,000-sample test set. With sklearn's default momentum enabled instead, the gap was 1.5 points (0.9791 vs. 0.9641) rather than 0.18 — most of the original gap was attributable to the optimizer mismatch, not to anything wrong with the custom implementation.

**Gradient check** (finite-difference vs. analytical, tiny synthetic network):

| Parameter | Mean Relative Error | Max Relative Error |
|---|---|---|
| W1 | 2.14e-10 | 1.49e-09 |
| W2 | 3.93e-10 | 7.39e-09 |
| W3 | 3.05e-10 | 2.75e-09 |
| b1 | 9.83e-11 | 4.19e-10 |
| b2 | 5.55e-11 | 2.05e-10 |
| b3 | 1.66e-11 | 2.93e-11 |

Overall max relative error `7.39e-09`, mean `2.62e-10` — every parameter matrix in the same tight band, which is itself part of the evidence: a genuine backprop bug tends to show up as one parameter family being wrong while the others are fine, not as six matrices all agreeing this closely.

**Hyperparameter sweep** (one factor at a time, 8-epoch budget unless noted):

| Group | Setting | Val. Accuracy | Val. Loss | Train Time |
|---|---|---|---|---|
| Learning rate | 0.001 | 0.8748 | 0.4580 | 22.3 s |
| Learning rate | 0.005 | 0.9232 | 0.2693 | 22.3 s |
| Learning rate | 0.01 (baseline) | 0.9413 | 0.2128 | 22.3 s |
| Learning rate | **0.05** | **0.9698** | **0.1050** | 23.6 s |
| Architecture | 64 | 0.9210 | 0.2785 | 12.3 s |
| Architecture | 128 | 0.9283 | 0.2665 | 19.2 s |
| Architecture | 128-64 (baseline) | 0.9413 | 0.2128 | 21.9 s |
| Architecture | **256-128** | **0.9457** | **0.1927** | 58.2 s |
| Batch size | **32** | **0.9567** | **0.1567** | 30.1 s |
| Batch size | 64 (baseline) | 0.9413 | 0.2128 | 22.6 s |
| Batch size | 128 | 0.9232 | 0.2693 | 17.8 s |
| Epochs | 10 | 0.9468 | 0.1939 | 29.1 s |
| Epochs | 20 (baseline, full budget) | 0.9593 | 0.1448 | 55.4 s |
| Epochs | **30** | **0.9647** | **0.1192** | 85.2 s |

**Ablations** (8-epoch budget):

| Experiment | Val. Accuracy | Notes |
|---|---|---|
| Single hidden layer (`784→128→10`) | 0.9283 | Identical to the `arch=128` sweep row by construction |
| Full-batch gradient descent | 0.1080 | Only 8 total parameter updates in this budget — essentially chance level, not "slower convergence" |
| Sigmoid hidden activations | 0.8368 | vs. 0.9413 for ReLU at the same 8-epoch budget |

**Error analysis:** the most-confused (true → predicted) digit pairs are `5→3` (23 cases), `9→4` (18), `4→9` (15), `6→5` (14), `9→3` (14), `7→2` (13), `8→3` (12), `6→4` (11), `7→1` (11), and `2→8` (9). Digit `3` has the lowest precision (0.939, the class most often mistaken for others), digit `9` has the lowest recall (0.937, the class most often missed — largely to `4` and `3`), and digit `1` is the easiest class (F1 = 0.982).

**Data pipeline:** 70,000 raw MNIST images (no missing values, no duplicate rows) → normalized to `[0, 1]` → split by MNIST's canonical convention into 60,000 train / 10,000 held-out test, with the training portion further split 90/10 (stratified) into 54,000 train / 6,000 validation. Class balance ratio (min/max count) = 0.801.

---

## What I Learned

1. **A "Generic" Class Is Only Generic If Its Forward and Backward Pass Actually Loop.**

   `__init__` built parameter matrices for an arbitrary `layer_sizes` list from the start, which made the class *look* general-purpose. But `forward()` and `backward()` were still written against a fixed three-weight-matrix mental model (`W1, W2, W3`), and that mismatch stayed invisible until the architecture sweep actually tried a shape the hardcoded version didn't anticipate. The fix wasn't just to patch the crash — it was to re-run gradient checking at multiple depths afterward, because a class that no longer raises a `KeyError` isn't the same claim as a class whose gradients are still correct.

2. **A Finite-Difference Check Can Fail for Reasons That Have Nothing to Do With the Gradient Being Wrong.**

   A relative error of `0.163` concentrated entirely in one bias matrix looked, at first glance, exactly like the signature of a real bug. Tracing it down to a single synthetic example landing precisely on a ReLU kink for every hidden-2 unit at once, purely a coincidence of zero-initialized biases and a tiny batch, was a reminder that a verification tool has its own failure modes, and that "isolated to one parameter family" versus "spread across everything" is itself diagnostic information worth reading before concluding anything.

3. **Two Models Reaching Similar Accuracy Doesn't Mean They Were Fairly Compared.**

   The custom MLP was already close to sklearn's `MLPClassifier` (a 1.5-point gap) before I'd checked what optimizer sklearn's default `solver="sgd"` actually runs, momentum-accelerated SGD, not the plain SGD the custom implementation uses. Once that one keyword argument was matched, the gap shrank to 0.18 points and the custom MLP occasionally edged ahead. The lesson wasn't "our model is good", it was that a "matched" comparison needs every default checked, not just the ones you remembered to set.

4. **Proving Correctness and Reporting Accuracy Are Different Claims.**

   A network can partially learn even with a subtly wrong backward pass, enough signal gets through to reduce loss without every gradient being right. Numerical gradient checking on a tiny, cheap-to-verify network is what actually certifies the math, independent of whatever accuracy the full-scale model reaches; treating a high accuracy number as sufficient proof of correctness would have been the weaker, more circumstantial argument.

5. **A Fair Benchmark Has to Say What Made It Unfair.**

   Training SVM and KNN on a smaller subset than the neural models makes any accuracy gap between them ambiguous, is the gap about the algorithm, or about the training-set size difference? Naming that training-set-size mismatch directly next to the results, rather than letting the table imply a clean apples-to-apples ranking, is what kept the benchmark honest. It also made Random Forest's 1st-place finish (on the *full* training set, unlike SVM/KNN/Gradient Boosting) a genuinely interpretable result rather than a confounded one.

6. **The Hypothesis You Write Down First Isn't Always the One the Data Confirms.**

   Going in, the expectation was that a tree ensemble would underperform the neural and kernel methods on dense pixel input. Random Forest instead finished 1st overall, ahead of both MLPs and the SVM. Writing the hypothesis down explicitly beforehand made it possible to recognize it had actually been falsified, and worth stating as a finding, rather than being an odd number quietly folded into a paragraph about "competitive classical baselines."

---

## Author

**Chinonso Franklin Nwankwo**
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)