"""Train the artifacts that stream.py loads.

Reproduces the pipeline from House_price_prediction.ipynb and writes:
  file.pkl   -- the fitted LinearRegression
  encod.pkl  -- the fitted OneHotEncoder

Run from anywhere:  python train.py
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "HousePricePrediction.xlsx"
MODEL_OUT = ROOT / "file.pkl"
ENCODER_OUT = ROOT / "encod.pkl"
RANDOM_STATE = 42
TRAIN_SIZE = 0.8


def load_data():
    df = pd.read_excel(DATA).drop(columns=["Id"])
    # Notebook behaviour: mean-fill the target, then drop rows with any other gap.
    df["SalePrice"] = df["SalePrice"].fillna(df["SalePrice"].mean())
    return df.dropna()


def encode(df):
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    one_hot = pd.DataFrame(
        encoder.fit_transform(df[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index,
    )
    return pd.concat([df.drop(columns=cat_cols), one_hot], axis=1), encoder


def main():
    df = load_data()
    encoded, encoder = encode(df)
    X = encoded.drop(columns=["SalePrice"])
    y = encoded["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )
    print(f"{len(df)} rows, {X.shape[1]} features after encoding")

    # The notebook scored every model against the Random Forest's predictions,
    # so its Linear Regression numbers were wrong. Each model is scored on its own here.
    candidates = {
        "SVR": SVR(),
        "RandomForest": RandomForestRegressor(
            n_estimators=10, random_state=RANDOM_STATE
        ),
        "LinearRegression": LinearRegression(),
    }
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, pred)
        print(f"  {name:17} MAPE {mape:7.4f}   R2 {r2_score(y_test, pred):7.3f}")

    chosen = candidates["LinearRegression"]  # the model stream.py expects
    MODEL_OUT.write_bytes(pickle.dumps(chosen))
    ENCODER_OUT.write_bytes(pickle.dumps(encoder))
    print(f"wrote {MODEL_OUT.name} and {ENCODER_OUT.name}")


if __name__ == "__main__":
    main()
