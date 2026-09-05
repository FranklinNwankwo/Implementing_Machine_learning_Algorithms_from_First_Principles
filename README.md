# Implementing Machine Learning Algorithms from First Principles

A collection of Machine Learning projects built from scratch using core scientific Python libraries (primarily NumPy). The goal is to develop a deep understanding of how these algorithms work under the hood, without relying on high-level abstractions like scikit-learn for the implementations themselves.

Each project follows the full ML pipeline: exploratory data analysis, feature engineering, from-scratch implementation, and validation against a reference library.

---

 ## SUPERVISED LEARNING

| # | Project | Algorithm | Dataset | Status |
|---|---|---|---|---|
| 01 | [Linear Regression](./1.%20Supervised_Learning/1.%20linear-regression_california_housing/) | OLS / Gradient Descent | California Housing | Complete |
| 02 | [Logistic Regression](./1.%20Supervised_Learning/2.%20logistic_regression_credit_card_fraud/) | Gradient Descent (Weighted BCE / SMOTE) | Credit Card Fraud Detection | Complete |
| 03 | [Decision Tree](./1.%20Supervised_Learning/3.%20decision_tree_breast_cancer/) | CART (Gini Impurity) | Breast Cancer Wisconsin | Complete |
| 04 | [Random Forest](./1.%20Supervised_Learning/4.%20random_forest_titanic_survival/) | Bagging + Random Feature Subsampling | Titanic Survival | Complete |
| 05 | [Gradient Boosting](./1.%20Supervised_Learning/5.%20gradient_boosting_adult_census_income/) | Functional Gradient Descent (Boosting) | Adult Census Income | Complete |
| 06 | [K-Nearest Neighbors](./1.%20Supervised_Learning/6.%20KNN_iris_flower/) | Brute-Force Distance Search (Majority Vote) | Iris Flower | Complete |
| 07 | [Naive Bayes](./1.%20Supervised_Learning/7.%20naive_bayes_newsgroups/) | Multinomial Naive Bayes (Bayes' Theorem) | 20 Newsgroups | Complete |
| 08 | [Support Vector Machine](./1.%20Supervised_Learning/8.%20SVM_breast_cancer/) | Soft-Margin Linear SVM (Subgradient Descent) | Breast Cancer Wisconsin | Complete |
| 09 | [Linear Discriminant Analysis](./1.%20Supervised_Learning/9.%20LDA_imdb_movie/) | Fisher's Criterion (Ridge-Regularised Pooled Covariance) | IMDb Movie Reviews | Complete |
| 10 | [Multilayer Perceptron](./1.%20Supervised_Learning/10.%20MLP_MNIST/) | Backpropagation (Mini-Batch Gradient Descent) | MNIST Handwritten Digits | Complete |


## UN-SUPERVISED LEARNING

| # | Project | Algorithm | Dataset | Status |
|---|---|---|---|---|


## REINFORCEMENT LEARNING

| # | Project | Algorithm | Dataset | Status |
|---|---|---|---|---|


---

## Structure

Each project lives in its own subdirectory and contains:

```
project_name/
├── data/            # Data files (usually not tracked; loaded programmatically)
├── notebook/        # Jupyter notebook
├── requirements.txt # Project-specific dependencies
└── README.md        # Project overview, results, and limitations
```

---

## Setup

Each project has its own `requirements.txt`. To get started with any project:

```bash
cd <project_folder>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebook/
```

---

## Author

**Franklin Nwankwo**  
[LinkedIn](https://www.linkedin.com/in/franklin-nwankwo-499736383/)
