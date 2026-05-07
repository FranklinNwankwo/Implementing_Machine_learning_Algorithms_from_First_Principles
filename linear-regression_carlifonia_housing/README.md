# California Housing — Linear Regression from Scratch

A end-to-end machine learning project implementing linear regression using only NumPy on the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset). 

Built to understand the fundamentals of the ML pipeline without relying on high-level abstractions.

---

## Project Overview

This project walks through the full supervised learning pipeline:

- **Exploratory Data Analysis (EDA)** — distributions, correlations, and outlier inspection
- **Feature Engineering** — transformations, interaction terms, and scaling
- **Model Implementation** — gradient descent and the normal equation built from scratch with NumPy
- **Validation** — results benchmarked against scikit-learn's `LinearRegression`
- **Diagnostics** — residual analysis, multicollinearity checks (VIF), and model limitations

Key issues encountered and resolved during the project:

- Data leakage (scaling before train/test split)
- Multicollinearity among features
- The dummy variable trap
- Rank deficiency in the design matrix

---

## Repository Structure

```
california-housing-regression/
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks
│   └── linear_regression.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

> **Note on data:** The raw dataset is not committed to this repo. It is loaded directly via this url: https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv inside the notebook, so no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/california-housing-regression.git
cd california-housing-regression
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
jupyter notebook notebooks/
```

---

## Dependencies

See `requirements.txt`. Core libraries used:

| Library | Purpose |
|---|---|
| `numpy` | Linear algebra, gradient descent |
| `pandas` | Data manipulation and EDA |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical plots |
| `scikit-learn` | For cross- validation baseline |
| `statsmodels` | VIF / multicollinearity diagnostics |
| `jupyter` | Notebook environment |

---

## Limitations

- Linear regression assumes a linear relationship between features and the target; the California Housing data has notable non-linearities.
- Outliers in `AveRooms` and `AveOccup` affect coefficient estimates.
- Geographic features (`Latitude`, `Longitude`) are used as raw inputs rather than being modelled spatially.
- No regularisation (Ridge/Lasso) is applied; the model may overfit on noisy features.

---

## Results

| Method      | Train R² | Test R² | Test RMSE |
|------------------|-----------|----------|------------|
| Normal Equation | 0.6743  | 0.4985  | $81,065  |
| Gradient Descent | 0.6707  | 0.4665  | $83,609  |
| Scikit-Learn   | 0.6743  | 0.4985  | $81,065  |

---

## What I learned:

1. The Full ML Pipeline Is More Than Just the Model.

Before writing a single line of model code, I had to think carefully about the entire pipeline: loading and inspecting raw data, splitting it into train and test sets, scaling features, engineering new ones, fitting the model, and then evaluating it. It sounds straightforward written out like that, but in practice each of those steps has subtle decisions embedded in it, and getting any one of them wrong can silently corrupt everything downstream. One of my biggest takeaways from this project is that the model itself is almost the least of your worries. The pipeline around it is where most of the complexity lives.

2. The Normal Equation: Closed-Form Linear Regression.

The first implementation I wrote used the Normal Equation: a closed-form solution that directly computes the optimal weights using matrix algebra. Deriving it from first principles, understanding that you're minimizing the sum of squared residuals and solving for the weight vector analytically was genuinely satisfying. It made linear regression feel like something I understood, not just something I'd read about.

The Normal Equation works well for smaller datasets, but it requires computing a matrix inverse, which becomes expensive as the number of features grows. This is part of why gradient descent exists as an alternative, which I implemented next.

3. Gradient Descent From Scratch.

Implementing gradient descent by hand was where things got really interesting. The update rule is simple enough on paper; subtract the gradient of the loss function scaled by a learning rate, but making it work reliably in practice involves several decisions: how to initialize weights, what learning rate to use, when to stop.
Getting it wrong in any of these areas produces models that diverge, oscillate, or just converge to the wrong place. I had to develop an intuition for what "healthy" training looks like (smooth, decreasing loss) versus what pathological training looks like. That intuition doesn't come from reading, but from from watching it break, and then improve iteratively.

4. Early Stopping

Rather than training for a fixed number of epochs, I implemented early stopping: monitoring the validation loss after each epoch and halting training when it stopped improving. This was my first exposure to the idea that more training isn't always better, and that generalization and training loss can diverge if you let the model train too long.

Early stopping also introduced the concept of "patience", waiting a certain number of epochs before giving up, to avoid stopping prematurely on a temporary plateau. Getting this right required careful tracking of the best weights seen so far and restoring them at the end of training, not just stopping at the last epoch.

5. Feature Engineering, and Why It's Dangerous

I added engineered features to try to improve model performance, including interaction terms and ratio features derived from the raw columns in the California Housing dataset. This immediately produced a wave of problems I didn't anticipate.

The Dummy Variable Trap. When encoding categorical variables with one-hot encoding, including all categories creates perfect multicollinearity; one column can be exactly predicted as a linear combination of the others. This makes the design matrix rank-deficient, and the Normal Equation breaks down because you can't invert a singular matrix. The fix is simple: drop one category, but you have to know to do it.

Rank Deficiency. Even beyond the dummy variable trap, adding too many engineered features (especially ones derived from each other) can reduce the rank of the feature matrix. I hit this, diagnosed it using np.linalg.matrix_rank, and had to carefully audit which features were redundant and remove them.

Multicollinearity. Even when the matrix is technically invertible, high correlation between features inflates the variance of coefficient estimates, making the model numerically unstable and the weights hard to interpret. Resolving this required checking correlation matrices, thinking carefully about which features were genuinely adding information, and removing those that were largely redundant.

6. Data Leakage: Small Mistake, Big Consequences

At one point in the project, I was fitting my StandardScaler on the entire dataset before splitting into train and test sets. This is data leakage: information from the test set is bleeding into the training process, making the model's evaluation metrics falsely optimistic.

The fix is trivial: fit the scaler only on the training set, then use that same fitted scaler to transform both train and test. But understanding why this matters took deliberate thought. The test set is supposed to simulate future, unseen data. If your preprocessing has already "seen" that data, you're not actually measuring generalization, you're measuring something closer to memorization of test statistics. This was a good lesson in the gap between code that runs and code that's correct.

7. Numerical Stability: Overflow Errors in Preprocessing

During feature engineering, I applied a log transformation to a feature that had already been transformed in a previous step. This caused overflow errors that crashed training entirely. It's the kind of bug that's easy to introduce when you're stacking preprocessing steps without keeping careful track of what state each feature is in at each stage.

The fix required me to audit the full transformation pipeline step by step, tracking what transformations had already been applied and making sure nothing was being double-transformed or applied to values outside its valid domain (like taking the log of a negative number or a zero). It reinforced the importance of being explicit and methodical about data state throughout the pipeline, not just at the beginning and end.

8. Benchmarking Against scikit-learn

Once I had a working model, I validated it by comparing its predictions against scikit-learn's LinearRegression function on the same data with the same features. The goal was to match exactly, not approximately, but to within floating-point rounding error.
Getting there required tracing through every discrepancy: mismatches in how features were scaled, differences in which transformations had been applied, subtle bugs in the Normal Equation implementation. But once everything aligned and the outputs matched, the validation felt airtight in a way that a standalone implementation never quite does. Using an established library as a ground-truth reference is something I'll carry into every future project.

9. Debugging Is Half the Work

If I had to summarize the meta-lesson from this project in one sentence: debugging is not what you do when things go wrong. It's a core part of the workflow. The model I ended up with is the product of multiple full debugging cycles, each one catching something the previous pass missed. Getting comfortable with that process, and building the habits to trace failures systematically rather than guessing, was probably the most transferable thing I took away from this project.

10. What I'd Do Differently

Looking back, there are several things I'd change if I were starting this project over. Not because the process was wrong, but because I now understand where I was flying blind and where a bit more structure upfront would have saved a lot of debugging time later.

Define the pipeline before writing any code: I built the pipeline somewhat organically: adding preprocessing steps, then features, then realizing something earlier was wrong and having to backtrack. Next time I'd sketch the full pipeline on paper first: what transformations happen in what order, what state each feature should be in at each stage, and where the train/test split boundary sits relative to every preprocessing step. That single diagram would have prevented the data leakage issue and the double-transformation overflow bug entirely.

Track feature state explicitly: A lot of my debugging time was spent figuring out whether a given feature had already been log-transformed, scaled, or had an interaction term derived from it. I'd solve this by being far more deliberate about naming conventions and maintaining a simple record of what had been applied to each column at each stage. It sounds tedious, but it's much less tedious than tracing an overflow error backwards through five preprocessing steps.

Validate incrementally, not just at the end: I compared against scikit-learn only after the model was "done." In hindsight, I should have been checking intermediate outputs much earlier; verifying that my scaled features matched sklearn's scaler output, that my design matrix matched what sklearn would produce, that my Normal Equation weights matched before even running gradient descent. Catching a discrepancy at the source is far cheaper than hunting it down at the end.

Be more deliberate about feature selection before adding complexity: I introduced engineered features somewhat aggressively before fully understanding the baseline model's behavior. The rank deficiency and multicollinearity issues that followed were a direct consequence. A cleaner approach would be to fully understand and validate the model on raw features first, then add engineered ones incrementally, checking the condition number and correlation matrix after each addition rather than all at once.

Write tests for preprocessing steps: The bugs that cost me the most time; leakage, double-transformations, rank deficiency, were all in the data pipeline, not in the model math. Simple assertions (e.g., confirming the scaler was fit only on training data, checking that no feature has been transformed twice, verifying the design matrix rank equals the number of features) would have caught most of them immediately. Treating preprocessing code with the same rigor as model code is something I'll prioritize going forward.

Document assumptions as I go: Several bugs were rooted in me forgetting an assumption I'd made earlier; that a certain feature had already been log-transformed, or that a scaler had been fit on a particular subset of the data. Writing those assumptions down inline, in the code, at the time I made them would have made the debugging process dramatically faster. Going forward, a comment explaining why a step is done a certain way is just as important as the step itself.


## Author

**Franklin Nwankwo**  
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)
