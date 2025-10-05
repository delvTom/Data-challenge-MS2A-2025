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

The competition evaluates models using a custom **weighted RMSE**,  
which penalizes more heavily errors on critical alarm levels (predicted ≥ 0.5).


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
📘 Notebook: [`02_baseline_random_forest.ipynb`](./notebooks/02_baseline_random_forest.ipynb)

---

### 3️⃣ **XGBRegressor – Randomized Search Optimization**  
- Implemented one **XGBRegressor per target variable**  
- Tuned hyperparameters with `RandomizedSearchCV`  
- Added **early stopping**, **regularization**, and clipping in [0,1]  
📘 Notebook: [`03_xgb_randomsearch.ipynb`](./notebooks/03_xgb_randomsearch.ipynb)

---

## 📈 Next Steps
- Integrate **adversarial validation** to detect train/test shifts  
- Add **feature engineering** (humidity percentile ranking, drift correction)  
- Experiment with **LightGBM** and **stacked ensemble methods**  
- Explore **regularization and Bayesian optimization**

---

## ⚙️ Tools and Libraries
- **Python**, **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**  
- **Scikit-learn**, **XGBoost**  
- **Google Colab**, **Git**, **LaTeX**

---

## 👤 Author

**Tom De Oliveira**  
Master 2 – Mathematics, Statistics and Learning (MS2A)  
Sorbonne Université, Paris  
📧 [tom.deoliveira@hotmail.fr](mailto:tom.deoliveira@hotmail.fr)

