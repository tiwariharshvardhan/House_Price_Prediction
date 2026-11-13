# 🏠 House Price Prediction

[![tests](https://github.com/tiwariharshvardhan/House_Price_Prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/tiwariharshvardhan/House_Price_Prediction/actions/workflows/tests.yml)

> Forked from [adarshagr-17/House_Price_Prediction](https://github.com/adarshagr-17/House_Price_Prediction).
> The original dataset, notebook, model and Streamlit app are by Adarsh Agrawal.
> This fork adds pinned dependencies, a reproducible training script and tests.

## 📖 Overview
This project focuses on predicting house prices using machine learning algorithms. By analyzing features such as the number of rooms, location, and size, the model provides accurate predictions, aiding in real estate decisions and market analysis.

## 💡 Features
Exploratory Data Analysis (EDA): Visualization and statistical analysis of the dataset to uncover insights.
Data Preprocessing: Handling missing values, feature encoding, and normalization.
Model Building: Training and evaluating regression models, including Linear Regression and other advanced algorithms.
Performance Metrics: Utilizing metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for evaluation.

## 📊 Dataset Description
The dataset includes:
Numerical Features: Area, number of rooms, price, etc.
Categorical Features: Location, property type, etc.
Target Variable: House price to be predicted.

## ⚙️ Methodology
Data Preprocessing:
Handle missing values using imputation techniques.
Normalize numerical features to bring them to a common scale.
Encode categorical variables using one-hot encoding or label encoding.
Model Training:
Train regression models to predict house prices.
Experiment with algorithms like Linear Regression, Decision Trees, and Random Forest.
Evaluation:
Use metrics like MAE, RMSE, and R² to assess model performance.
Fine-tune hyperparameters to improve prediction accuracy.

## 🛠️ Tools and Technologies
Programming Language: Python
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
Environment: Jupyter Notebook

## 🚀 Running it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run stream.py
```

Retrain the model and encoder from the spreadsheet:

```bash
.venv/bin/python train.py
```

Run the tests:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

## 📈 Results
Scored on a held-out 20% split (2,913 rows after cleaning, 37 features after encoding):

| Model | MAPE | R² |
| --- | --- | --- |
| Linear Regression | 0.200 | 0.347 |
| Random Forest (10 trees) | 0.206 | 0.329 |
| SVR | 0.207 | -0.001 |

Linear Regression wins, and it is the model saved to `file.pkl` and served by the app.

An R² of 0.35 with a mean error near 20% makes this a rough estimate, not a valuation.
The dataset carries only 11 features and nothing about location quality or living area,
which is most of what actually moves a sale price.

Note on the numbers: the original notebook scored every model against the Random
Forest's predictions, so the figures it printed under Linear Regression were not its
own. `train.py` scores each model separately, and the table above is the result.
