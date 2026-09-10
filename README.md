# Implementing Machine Learning Algorithms from First Principles

A collection of Machine Learning projects built from scratch using core scientific Python libraries (primarily NumPy). The goal is to develop a deep understanding of how these algorithms work under the hood, without relying on high-level abstractions like scikit-learn for the implementations themselves.

Each project follows the full ML pipeline: exploratory data analysis, feature engineering, from-scratch implementation, and validation against a reference library.

---

 ## SUPERVISED LEARNING

| # | Project | Algorithm | Dataset | Status |
|---|---|---|---|---|
| 01 | [Linear Regression](./1.%20Supervised_Learning/01.%20linear-regression_california_housing/) | OLS / Gradient Descent | California Housing | Complete |
| 02 | [Logistic Regression](./1.%20Supervised_Learning/02.%20logistic_regression_credit_card_fraud/) | Gradient Descent (Weighted BCE / SMOTE) | Credit Card Fraud Detection | Complete |
| 03 | [Decision Tree](./1.%20Supervised_Learning/03.%20decision_tree_breast_cancer/) | CART (Gini Impurity) | Breast Cancer Wisconsin | Complete |
| 04 | [Random Forest](./1.%20Supervised_Learning/04.%20random_forest_titanic_survival/) | Bagging + Random Feature Subsampling | Titanic Survival | Complete |
| 05 | [Gradient Boosting](./1.%20Supervised_Learning/05.%20gradient_boosting_adult_census_income/) | Functional Gradient Descent (Boosting) | Adult Census Income | Complete |
| 06 | [K-Nearest Neighbors](./1.%20Supervised_Learning/06.%20KNN_iris_flower/) | Brute-Force Distance Search (Majority Vote) | Iris Flower | Complete |
| 07 | [Naive Bayes](./1.%20Supervised_Learning/07.%20naive_bayes_newsgroups/) | Multinomial Naive Bayes (Bayes' Theorem) | 20 Newsgroups | Complete |
| 08 | [Support Vector Machine](./1.%20Supervised_Learning/08.%20SVM_breast_cancer/) | Soft-Margin Linear SVM (Subgradient Descent) | Breast Cancer Wisconsin | Complete |
| 09 | [Linear Discriminant Analysis](./1.%20Supervised_Learning/09.%20LDA_imdb_movie/) | Fisher's Criterion (Ridge-Regularised Pooled Covariance) | IMDb Movie Reviews | Complete |
| 10 | [Multilayer Perceptron](./1.%20Supervised_Learning/10.%20MLP_MNIST/) | Backpropagation (Mini-Batch Gradient Descent) | MNIST Handwritten Digits | Complete |


## UN-SUPERVISED LEARNING

| # | Project | Algorithm | Dataset | Status |
|---|---|---|---|---|
| 1 | [K-Means Clustering](./2.%20Unsupervised_Learning/1.%20K-Means_Iris_flower/) | Custom Centroid Assignment & Update (Lloyd's Algorithm) | Iris Flower Dataset | Complete |
|2 | [Hierarchical Clustering](./2.%20Unsupervised_Learning/2.%20Hierarchical_Clustering_mall_dataset/) | Custom Agglomerative Clustering (Ward Linkage via Lance-Williams Update) | Mall Customers Dataset | Complete|
|3 | [Principal Component Analysis](./2.%20Unsupervised_Learning/3.%20PCA_wine_dataset/) | Custom PCA (Eigendecomposition via NumPy `eigh`) | Wine Dataset | Complete|
|4 | [DBSCAN Clustering](./2.%20Unsupervised_Learning/4.%20DBSCAN_online_retail_dataset/) | Custom DBSCAN (Region Queries & Density-Reachability via Epsilon-Neighborhoods) | UCI Online Retail Dataset | Complete|
|5 | [Gaussian Mixture Model](./2.%20Unsupervised_Learning/5.%20Gaussian_Mixture_Model_olivetti_faces/) | Custom GMM (Expectation-Maximization via NumPy/SciPy, Cholesky-based log-densities) | Olivetti Faces Dataset | Complete|

## REINFORCEMENT LEARNING

| # | Project | Algorithm | Environment | Status |
|---|---|---|---|---|
| 1 | [Q-Learning](3.%20Reinforcement_Learning/1.%20Q-Learning_FrozenLake) | Q-Learning (Tabular, from scratch) | FrozenLake-v1 (Gymnasium) | Complete |

---

## Structure

Each project lives in its own subdirectory and contains:

```
project_name/
├── data/            # Data files (usually not tracked; loaded programmatically)
├── notebook/        # Jupyter notebook
├── README.md        # Project overview, results, and limitations
└── requirements.txt # Project-specific dependencies
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

**Chinonso Franklin Nwankwo**  
[LinkedIn](https://www.linkedin.com/in/chinonso-nwankwo/)
