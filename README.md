# Data Challenge 2025 – Toxic Gas Detection  
*(ENS Challenge Data x Bertin Technologies)*

This repository contains my work for the **ENS Challenge Data 2025**, organized in collaboration with **Bertin Technologies**.  
The goal is to **predict alarm levels for 23 categories of toxic gases** (e.g., vesicants, sarin, etc.) based on industrial sensor data.  
It is a **multi-output regression** task that evaluates both prediction accuracy and robustness.

---

## 🧠 Challenge Overview

The dataset contains about **330,000 samples** describing environmental and sensor-based variables.  
Each sample corresponds to a measurement, and the targets \((c_1, ..., c_{23})\) represent **alarm levels** between 0 and 1 for each gas type.  

The challenge focuses on:
- Predicting alarm intensities for multiple correlated gases  
- Handling high-dimensional, noisy sensor data  
- Managing **distribution shifts** between training and testing sets  
- Designing efficient and interpretable regression models  

---

## 📏 Evaluation Metric – Root Weighted Mean Squared Error (RW-MSE)

The evaluation metric is a **root mean squared error with per-target weighting** that penalizes underestimation of critical alarms.

For each sample, the weighted squared error is:

\[
E = \frac{1}{23} \sum_{i=1}^{23} f_i (c_i - \hat{c}_i)^2
\]

where the weights \(f_i\) are defined as:

\[
f_i =
\begin{cases}
1 & \text{if } \hat{c}_i < 0.5 \\
1.2 & \text{if } \hat{c}_i \ge 0.5
\end{cases}
\]

The final score is the square root of the mean error across all samples:

\[
\text{Score} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} E_n}
\]

➡️ This formulation gives more importance to cases where an alarm should be triggered (predicted level ≥ 0.5).

---

## 🚧 Current Progress (October 2025)

The project currently includes three main steps:

### 1️⃣ **Exploratory Data Analysis (EDA)**  
- Duplicate and missing value removal  
- Outlier and feature distribution analysis  
- Correlation study between features and targets  
- Humidity variable cleaning and normalization  
📘 Notebook: [`01_eda.ipynb`](./notebooks/01_eda.ipynb)

---

### 2️⃣ **Baseline Model – Random Forest**  
- Implemented `RandomForestRegressor` with `MultiOutputRegressor`  
- Established a first baseline with **public score ≈ 0.1523 (RW-MSE)**  
📘 Notebook: [`02_baseline_rand]()

---

## 📏 Evaluation Metric – Weighted MAE

The competition evaluates models using a **weighted Mean Absolute Error (wMAE)** across all target variables:

\[
wMAE = \frac{\sum_{j=1}^{23} w_j \cdot MAE_j}{\sum_{j=1}^{23} w_j}
\]

where  
\[
MAE_j = \frac{1}{N} \sum_{i=1}^{N} |y_{ij} - \hat{y}_{ij}|
\]  
and \( w_j \) is the weight associated with target \( j \).

This ensures that more critical gas types have a stronger influence on the final score.

---

## 🚧 Current Progress (October 2025)

The project is currently in progress and includes three main stages:

### 1️⃣ **Exploratory Data Analysis (EDA)**  
- Duplicate and missing value handling  
- Outlier and feature distribution analysis  
- Correlation study between features and targets  
- Basic target normalization and scaling  
📘 Notebook: [`01_eda.ipynb`](./notebooks/01_eda.ipynb)

---

### 2️⃣ **Baseline Model – Random Forest**  
- Implemented `RandomForestRegressor` with `MultiOutputRegressor`  
- Established first baseline with public leaderboard score ≈ **0.1523 (wMAE)**  
📘 Notebook: [`02_baseline_random_forest.ipynb`](./notebooks/02_baseline_random_forest.ipynb)

---

### 3️⃣ **XGBRegressor – Randomized Search Optimization**  
- Implemented individual **XGBRegressor per target**  
- Hyperparameter tuning with `RandomizedSearchCV`  
- Added **early stopping** and **regularization** to improve generalization  
📘 Notebook: [`03_xgb_randomsearch.ipynb`](./notebooks/03_xgb_randomsearch.ipynb)

---

## 📈 Next Steps
- Integrate **adversarial validation** to handle train/test drift  
- Experiment with **feature engineering** (humidity normalization, percentile clipping)  
- Add **LightGBM** and **stacking/blending** approaches  
- Perform **cross-validation** and finalize model selection  

---

## ⚙️ Tools and Libraries
- **Python**, **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**  
- **Scikit-learn**, **XGBoost**  
- **Google Colab**, **GitHub**

---

## 👤 Author

**Tom De Oliveira**  
Master 2 – Mathematics, Statistics and Learning (MS2A)  
Sorbonne Université, Paris  
📧 [tom.deoliveira@hotmail.fr](mailto:tom.deoliveira@hotmail.fr)
