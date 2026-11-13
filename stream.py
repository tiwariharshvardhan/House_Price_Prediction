import streamlit as st
import pickle
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    """Unpickle the model and encoder once per session, not on every rerun."""
    try:
        with open(ROOT / "file.pkl", "rb") as f:
            classifier = pickle.load(f)
        with open(ROOT / "encod.pkl", "rb") as f:
            encoder = pickle.load(f)
    except FileNotFoundError as err:
        st.error(
            f"Could not find `{Path(err.filename).name}`. "
            "Run `python train.py` to build the model files, then reload."
        )
        st.stop()
    return classifier, encoder


def predict_price(features):
    classifier, encoder = load_artifacts()
    # Ask the encoder which columns it was fitted on rather than sniffing dtypes,
    # which pandas 4 changes for object/str columns.
    cat_cols = list(encoder.feature_names_in_)

    one_hot = pd.DataFrame(
        encoder.transform(features[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=features.index,
    )
    encoded = pd.concat([features.drop(columns=cat_cols), one_hot], axis=1)

    # Same column order the model was trained with; anything absent is a zero.
    encoded = encoded.reindex(columns=classifier.feature_names_in_, fill_value=0)
    return classifier.predict(encoded)

def prediction_form():
    _, encoder = load_artifacts()  # surface a missing-model error before the form, not after
    # Offer exactly the categories the encoder was fitted on, so the two can't drift apart.
    options = dict(zip(encoder.feature_names_in_, encoder.categories_))
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">House Price Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    # Categorical inputs
    MSZoning = st.selectbox("MSZoning", options["MSZoning"])
    LotConfig = st.selectbox("LotConfig", options["LotConfig"])
    BldgType = st.selectbox("BldgType", options["BldgType"])
    Exterior1st = st.selectbox("Exterior1st", options["Exterior1st"])
    # Numerical Inputs (Convert to float/int)
    MSSubClass = int(st.number_input("MSSubClass", min_value=0, max_value=200, value=60))
    LotArea = float(st.number_input("LotArea", min_value=1000, max_value=50000, value=7000))
    OverallCond = int(st.number_input("OverallCond", min_value=1, max_value=10, value=5))
    YearBuilt = int(st.number_input("YearBuilt", min_value=1800, max_value=2025, value=2000))
    YearRemodAdd = int(st.number_input("YearRemodAdd", min_value=1800, max_value=2025, value=2010))
    BsmtFinSF2 = float(st.number_input("BsmtFinSF2", min_value=0, max_value=2000, value=0))
    TotalBsmtSF = float(st.number_input("TotalBsmtSF", min_value=0, max_value=5000, value=1000))
    # Convert input into DataFrame
    input_data = pd.DataFrame({
        "MSSubClass": [MSSubClass],
        "MSZoning": [MSZoning],
        "LotArea": [LotArea],
        "LotConfig": [LotConfig],
        "BldgType": [BldgType],
        "OverallCond": [OverallCond],
        "YearBuilt": [YearBuilt],
        "YearRemodAdd": [YearRemodAdd],
        "Exterior1st": [Exterior1st],
        "BsmtFinSF2": [BsmtFinSF2],
        "TotalBsmtSF": [TotalBsmtSF]
        })
    if st.button("Predict"):
        price = predict_price(input_data)[0]
        st.success(f"Estimated sale price: ${price:,.0f}")


def main():
    st.set_page_config(page_title="Streamlit App", layout="wide")
    st.title("House Price Prediction")
    menu = ["Home", "About", "Contact"]
    choice = st.sidebar.selectbox("Navigation", menu)
    if choice == "Home":
        st.header("Home Page")
        st.write("This is the home page of the app.")
        prediction_form()
    
    elif choice == "About":
        st.header("About Page")
        st.write("This page contains information about the app.")
        
    elif choice == "Contact":
        st.header("Contact Page")
        st.write("This page contains contact details.")
    
if __name__ == "__main__":
    main()
