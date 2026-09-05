# Credit Card Fraud Detection — Logistic Regression from Scratch

An end-to-end machine learning project implementing logistic regression using only NumPy on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle — ULB Machine Learning Group).

Built to understand the fundamentals of classification under severe class imbalance without relying on high-level abstractions.

---

## Project Overview

This project walks through the full supervised learning pipeline for an extremely imbalanced binary classification problem. The raw dataset contains 284,807 transactions; after dropping 1,081 duplicate rows, 283,726 remain, with 473 confirmed fraud cases, a **598.8 : 1** legitimate-to-fraud ratio (fraud rate ≈ 0.167%):

- **Exploratory Data Analysis (EDA)** — class distribution, feature distributions by class, distributional overlap
- **Data Handling** — stratified three-way split, Z-score normalisation fit on train only, log-transform of `Amount`
- **Model Implementation** — sigmoid, weighted binary cross-entropy, gradient descent, and SMOTE all built from scratch with NumPy
- **Imbalance Handling** — two independent strategies compared: class-weighted loss (Model A) vs. SMOTE oversampling (Model B)
- **Threshold Optimisation** — F-beta (β=2) sweep on the validation set, since the default τ=0.5 fails under this imbalance
- **Validation** — results benchmarked against scikit-learn's `LogisticRegression`, Decision Tree, Random Forest, and XGBoost
- **Diagnostics** — learning curves, calibration (reliability diagrams + Brier score), error analysis, feature weight analysis, assumption audit
- **Gradient Verification** — analytic gradient checked against finite-difference approximation (relative error 3.46e-11) before any training

Key issues encountered and resolved during the project:

- Choosing evaluation metrics that don't lie under 99.84% majority-class accuracy
- Data leakage risk from scaling, SMOTE, or threshold tuning touching validation/test data
- Class imbalance collapsing gradient descent into predicting the majority class
- Miscalibrated probabilities produced by class weighting and oversampling
- Distinguishing a genuine cost/precision trade-off from a false claim of "irrecoverable" errors (see Performance Ceiling in Results)
- Brier score being a misleading top-line calibration metric at this base rate (see Results)

---

## Repository Structure

```
2. logistic_regression_credit_card_fraud/
├── data/
│ └── .gitkeep
├── notebook/
│ └── Logistic_Regression_Credit_Card_Dataset.ipynb
├── README.md
└── requirements.txt
```


> **Note on data:** The raw dataset is not committed to this repo. It is loaded directly via this url: https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv inside the notebook, so no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/FranklinNwankwo/Implementing_Machine_learning_Algorithms_from_First_Principles.git
cd "Implementing_Machine_learning_Algorithms_from_First_Principles/1. Supervised_Learning/2. logistic_regression_credit_card_fraud"
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
| `numpy` | Linear algebra, gradient descent, SMOTE, all metrics |
| `pandas` | Data manipulation and EDA |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical plots |
| `scipy` | KDE for distribution plots |
| `xgboost` | Tree-based benchmark comparison |
| `scikit-learn` | Stratified splitting and benchmark baselines |

---

## Limitations

- **Linear separability.** Both from-scratch models need an almost-zero threshold (τ≈0.003–0.007) to catch every single fraud case in the test set, at a precision of ~0.002, essentially flagging everyone as fraud. That's a real cost/precision limitation, but not literal irrecoverability: `sigmoid(z) > 0` for every finite `z`, so recall trivially reaches 1.0 for *any* model at a low enough threshold. The tree-based benchmarks (Random Forest, XGBoost outperforming both LR variants on F2/AUC-PR) are the actual controlled evidence that a nonlinear boundary helps here.
- **PCA information ceiling**: `V1`–`V28` were PCA-transformed upstream by the dataset creators. Any fraud-relevant signal discarded in that transformation is permanently invisible to any model trained on this data, and the resulting weights can't be mapped back to business-meaningful features.
- **SMOTE interpolates blindly**: synthetic samples are generated between any two minority-class neighbours regardless of whether the interpolated point is realistic or falls near a class boundary.
- **48-hour dataset window**: fraud patterns evolve over time; no concept-drift mechanism is implemented.
- **Independence assumption untestable**: there is no cardholder ID, so within-cardholder correlation across multiple transactions can't be measured or corrected for.
- **No explicit regularisation**: L2 weight decay was not implemented on the from-scratch model; early stopping is its sole overfitting control. This is also the likely reason its AUC-PR outpaces sklearn's L2-regularised LR (see Results).
- **Brier score is not a fair calibration bar at this prevalence.** Predicting p≈0 for every transaction is "correct" 99.84% of the time by construction, so a majority-class dummy trivially posts a near-zero Brier score with zero actual fraud-detection skill. Both from-scratch models score *worse* than the dummy on raw Brier (see Results); that reflects the metric's blind spot at extreme imbalance, not a calibration failure; AUC-ROC/AUC-PR/F2 are the metrics that actually reflect fraud-detection skill here.

---

## Results

Values below are the actual outputs of the notebook (`SEED=42`), taken directly from the executed evaluation cells, not estimates.

**Dummy baseline** (always predicts legitimate): Accuracy 99.84%, Precision 0.0000, Recall 0.0000, F2 0.0000, TP 0, FN 70, FP 0, AUC-PR ≈ 0.0017 (the fraud base rate — the no-skill floor).

| Model | τ | Precision | Recall | F2 | AUC-PR | AUC-ROC | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| **LR A — Weighted BCE (from scratch)** | 0.987 | 0.6667 | 0.8286 | 0.7902 | 0.5578 | **0.9770** | 58 | 12 | 29 |
| **LR B — SMOTE (from scratch)** | 0.767 | 0.6667 | 0.8286 | 0.7902 | 0.7548 | 0.9750 | 58 | 12 | 29 |
| sklearn LogisticRegression | 0.990 | 0.6105 | 0.8286 | 0.7733 | 0.2897 | 0.9783 | 58 | 12 | 37 |
| Decision Tree | 0.990 | 0.6292 | 0.8000 | 0.7588 | 0.6619 | 0.8871 | 56 | 14 | 33 |
| Random Forest | 0.626 | 0.7973 | **0.8429** | 0.8333 | 0.7810 | **0.9795** | 59 | 11 | 15 |
| **XGBoost** | 0.797 | **0.9661** | 0.8143 | **0.8407** | **0.8354** | 0.9487 | 57 | 13 | **2** |

**Model A and Model B tie exactly on Precision/Recall/F2/TP/FN/FP** at their respective optimal thresholds, a genuine coincidence, not a bug (they score visibly differently from each other on the validation-set threshold sweep earlier in the notebook, and their full Precision-Recall curves differ). With only 70 fraud cases in the test set, point-estimate metrics like these have limited resolution and shouldn't be over-read either way.

**AUC-PR — computed across the full threshold range rather than one operating point — is the metric that actually separates the two models, and Model B wins it clearly** (0.7548 vs 0.5578). The threshold story is still informative on its own: Model A needs τ≈0.987 to hit its optimal point, it only flags fraud when almost certain, while Model B reaches the *same* precision/recall at a meaningfully lower τ≈0.767, consistent with SMOTE giving the decision boundary richer signal in the fraud region rather than relying purely on reweighted loss.

**Both from-scratch models beat sklearn's LogisticRegression on AUC-PR** (0.5578 / 0.7548 vs. 0.2897) despite similar recall and F2, most likely because sklearn's default L2 regularisation pulls its probability estimates toward more conservative values, while the from-scratch model has no regularisation term — AUC-PR, being threshold-independent and rank-sensitive, rewards the wider probability spread this produces. See Limitations for the corresponding gap this leaves in the from-scratch model.

**XGBoost is the strongest model overall** — highest F2 (0.8407) and highest AUC-PR (0.8354), plus near-perfect precision (0.9661) with only 2 false alarms, at the cost of somewhat lower recall than Random Forest. **Random Forest is close behind and takes the highest recall (0.8429) and highest AUC-ROC (0.9795).** Both tree ensembles outperform both logistic regression variants on F2 and AUC-PR — real, controlled evidence (identical pipeline, identical metrics) that a nonlinear decision boundary captures fraud patterns a hyperplane can't fully reach, though **Model B's own AUC-PR (0.7548) already lands in the same range as Random Forest's (0.7810)** — the gap between "best linear" and "best nonlinear" here is real but not enormous.

**The dummy row is intentional.** 99.84% accuracy while catching zero fraud is the clearest possible illustration of why accuracy is the wrong metric for this problem.


### Calibration — Brier score

| | Brier score (lower = less raw squared error) |
|---|---|
| Dummy baseline | 0.001645 |
| Model A (Weighted BCE) | 0.025058 |
| Model B (SMOTE) | 0.002776 |

Both models score **worse** than the dummy on raw Brier score. This is expected, not a calibration failure: at 0.167% prevalence, predicting p≈0 for everyone is "correct" 99.84% of the time by construction, so the dummy gets an artificially tiny Brier score with zero actual fraud-detection skill. Beating it isn't a meaningful bar here — AUC-ROC (0.977 / 0.975), AUC-PR (0.558 / 0.755), and F2 (0.790 / 0.790) are the metrics that reflect real fraud-detection skill; Brier score at this imbalance mainly measures how "confidently correct" a classifier is about the overwhelming legitimate majority, which a fraud-focused model with extreme class weighting or oversampling is not optimising for.

---

## What I learned:

1. **Accuracy Is Actively Misleading Here.**

The first real lesson of this project had nothing to do with modelling, it was about metric selection. A classifier that predicts "legitimate" for every single transaction hits 99.84% accuracy while catching zero fraud. Building that dummy baseline explicitly, rather than skipping straight to the real model, made it viscerally clear why Recall, F2-score, and PR-AUC have to be the primary metrics under this kind of imbalance, and why accuracy has to be actively excluded, not just deprioritized.

2. **Weighted Loss vs. Oversampling Are Genuinely Different Interventions — Even When They Tie on Point Metrics.**

Going in, I expected class-weighted BCE and SMOTE to be roughly interchangeable fixes for imbalance. The surprise was that they landed on *identical* Precision/Recall/F2/confusion matrices at their respective optimal thresholds (58 TP / 12 FN / 29 FP for both), a coincidence given how differently they behave on the validation-set threshold sweep, and a reminder that with only 70 test-set fraud cases, point-estimate metrics have limited resolution. What actually separated them was AUC-PR, computed across the whole threshold range rather than one operating point: Model B (SMOTE) at 0.7548 clearly ahead of Model A (Weighted BCE) at 0.5578. They also reached their tied operating points at very different thresholds (τ≈0.987 vs τ≈0.767), evidence that SMOTE's synthetic points genuinely reshape the probability landscape rather than just rescaling it the way class weighting does.

3. **The Default Threshold of 0.5 Is an Assumption, Not a Law.**

τ=0.5 implicitly assumes false positives and false negatives cost the same. In fraud detection they don't. A missed fraud is a direct financial loss, a false alarm is customer friction. Implementing an F-beta (β=2) sweep on the validation set to explicitly weight recall over precision, rather than accepting the sklearn default, was the point where threshold selection stopped feeling like an afterthought and started feeling like a first-class modelling decision.

4. **A High AUC Doesn't Mean the Probabilities Are Trustworthy, and Even the "Trustworthy" Check Needs Checking.**

Both models score AUC-ROC above 0.97, which looks great in isolation and would have been misleading to stop at. Quantile-based bins were necessary for the reliability diagrams instead of equal-width ones, since equal-width bins are mostly empty at 0.17% fraud prevalence. But the calibration story went a step further than expected: raw Brier score actually favours the do-nothing dummy classifier (0.0016) over both real models (0.0251 / 0.0028), which at first reads like a red flag. It isn't, it's an artifact of Brier score at extreme imbalance (predicting "never fraud" is right 99.84% of the time), and it was a useful lesson in not trusting a single calibration number without understanding what it's actually sensitive to.

5. **Gradient Checking Is Worth Doing Even When the Math Feels Obvious.**

Before trusting any training run, I verified the analytic gradient against a finite-difference approximation and required the relative error to fall below 1e-5 (it came in at 3.46e-11). It's a debugging habit borrowed from deep learning, and it caught the class of bug that's otherwise nearly invisible: a model that trains, converges, and produces plausible-looking numbers while being subtly wrong underneath.

6. **Interpretability Has a Hard Ceiling Independent of the Model.**

`V1`–`V28` are anonymised PCA components the dataset creators applied PCA before I ever saw the data, specifically to protect cardholder privacy. That means the learned weights can tell me which components the model relies on, but never what those components mean in business terms. No amount of model sophistication recovers that; it's a property of the data, not of logistic regression specifically.

7. **A Threshold Sweep Can't Prove "Fundamental" Errors — and I Initially Got This Wrong.**

An earlier version of this project swept recall down to τ=0.01 and concluded that some fraud cases were permanently misclassified "regardless of threshold" — treating that as proof of a hard linear-boundary ceiling. That conclusion doesn't hold: since every model's sigmoid output is strictly positive, recall trivially reaches 1.0 at τ→0 for *any* classifier, so a threshold sweep alone can never show an irrecoverable error. The corrected, honest version of the diagnostic asks how *high* τ can go while recall stays at 1.0 (τ≈0.003–0.007 here) and what precision that costs (≈0.002) — a real but narrower finding about the cost of chasing the hardest cases, not about the geometry of the decision boundary. The actual, controlled evidence that nonlinearity helps here is the benchmark table: Random Forest and XGBoost outperform both logistic regression variants on F2 and AUC-PR (0.8333/0.8407 vs. 0.7902 for both LR models) under the identical pipeline and metrics. This was a good lesson in checking whether a diagnostic actually tests the claim it's being used to support, rather than trusting a plausible-sounding narrative around a real number.

8. **Numerical Stability Details Compound Silently.**

Sigmoid clipped to [-500, 500] to prevent overflow, BCE guarded with ε=1e-15 to prevent log(0), Z-score normalisation guarded with ε=1e-8 to prevent division by zero. Each of these felt like a minor detail in isolation, but skipping any one of them would have produced NaNs or silently wrong gradients somewhere downstream, likely long after the point where the actual bug was introduced.

9. **Benchmarking Against sklearn Is the Real Validation, And It Surfaced a Real Discrepancy.**

Once the from-scratch model was working, I checked it against sklearn's `LogisticRegression` on the same preprocessed data. AUC-ROC and F2 lined up closely (0.9770 vs 0.9783, 0.7902 vs 0.7733), which confirmed the core math was correct. But AUC-PR did not line up (0.5578 vs 0.2897), a genuine, sizeable gap rather than solver noise. Tracing that back to sklearn's default L2 regularisation (which the from-scratch model doesn't implement) was more valuable than the implementation itself, because it's the only way to be confident the from-scratch math is actually correct and to understand *why* a discrepancy exists rather than dismissing it.

10. **A Correctness Pass Can Move the Numbers More Than the Modelling Does.**

Going back through this notebook for bugs after treating it as "done" changed more than cosmetics. Two implementation bugs (a SMOTE target-ratio formula that under-shot its stated goal, and a Precision-Recall-curve convention that understated AUC-PR at the low-recall end) shifted AUC-PR materially — most visibly for Model B (0.4682 → 0.7548) and for every tree-based benchmark (e.g. XGBoost 0.4425 → 0.8354), which flipped the "best model" conclusion from Random Forest to XGBoost. A results-comparison table that silently marked the wrong column as the winner, and a Brier-score interpretation that asserted a conclusion the printed numbers didn't support, were both bugs I would have shipped past without a dedicated correctness review. The lesson: re-reading code for logic is not the same activity as re-checking whether the printed narrative actually follows from the printed numbers.

11. **What I'd Do Differently**

Looking back, a few things would have saved time. I'd write the leakage-prevention rules (what gets fit on train-only, what stays unseen until final evaluation) down explicitly before writing any preprocessing code, rather than re-deriving them mid-implementation. I'd also build the calibration diagnostics earlier in the process instead of near the end, discovering that Model A's probabilities were compressed toward the extremes after already comparing both models on threshold behaviour meant re-interpreting earlier conclusions in light of a fact I could have known from the start. I'd also match sklearn's regularisation (or explicitly disable it) from the start, so the benchmark comparison isolates "from-scratch vs. sklearn implementation correctness" instead of quietly also comparing "regularised vs. unregularised." And I'd treat every printed "conclusion" sentence as a claim to verify against its own numbers before considering a notebook finished, not just the code that produced the numbers.

---

## Author

**Chinonso Franklin Nwankwo**  
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)