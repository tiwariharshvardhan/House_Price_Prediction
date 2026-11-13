import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder


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


def predict_price(unput):
    classifier, encoder = load_artifacts()
    # Identify categorical columns
    obj_col = unput.select_dtypes(include=['object']).columns.tolist()

    # Transform categorical columns using the pre-fitted encoder
    one_hot_encoded = encoder.transform(unput[obj_col])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(obj_col), index=unput.index)

    # Drop original categorical columns and merge with encoded features
    df_encoded = unput.drop(columns=obj_col)
    df_encoded = pd.concat([df_encoded, one_hot_df], axis=1)

    # Ensure the column order matches training
    expected_columns = classifier.feature_names_in_
    df_encoded = df_encoded.reindex(columns=expected_columns, fill_value=0)  # Fill missing columns with 0

    # Predict using the classifier
    prediction = classifier.predict(df_encoded)
    return prediction

def hommie():
    load_artifacts()  # surface a missing-model error before the form, not after
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">House Price Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    # Categorical Inputs (Convert to String)
    MSZoning = st.selectbox("MSZoning", {"RL":'MSZoning_RL', "RM":'MSZoning_RM', "C (all)":'MSZoning_C (all)', "FV":'MSZoning_FV', "RH":'MSZoning_RH'})  # Example categories
    LotConfig = st.selectbox("LotConfig", {"Inside":'LotConfig_Inside', "Corner":'LotConfig_Corner', "CulDSac":'LotConfig_CulDSac', "FR2":'LotConfig_FR2', "FR3":'LotConfig_FR3'})
    BldgType = st.selectbox("BldgType", {"1Fam":'BldgType_1Fam', "2fmCon":'BldgType_2fmCon', "Duplex":'BldgType_Duplex', "Twnhs":'BldgType_Twnhs', "TwnhsE":'BldgType_TwnhsE'})
    Exterior1st = st.selectbox("Exterior1st", {"AsbShng":'Exterior1st_AsbShng',"AsphShn":'Exterior1st_AsphShn',"BrkComm":'Exterior1st_BrkComm',"BrkFace":'Exterior1st_BrkFace',"CBlock":'Exterior1st_CBlock',"CemntBd":'Exterior1st_CemntBd',"HdBoard":'Exterior1st_HdBoard',"ImStucc":'Exterior1st_ImStucc',"MetalSd":'Exterior1st_MetalSd',"Plywood":'Exterior1st_Plywood',"Stone":'Exterior1st_Stone',"Stucco":'Exterior1st_Stucco',"VinylSd":'Exterior1st_VinylSd',"Wd Sdng":'Exterior1st_Wd Sdng',"WdShing":'Exterior1st_WdShing'})
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
    result=""
    if st.button("Predict"):
        result=predict_price(input_data)
    st.success('The output is {}'.format(result))


def main():
    st.set_page_config(page_title="Streamlit App", layout="wide")
    st.title("House Price Prediction")
    menu = ["Home", "About", "Contact"]
    choice = st.sidebar.selectbox("Navigation", menu)
    if choice == "Home":
        st.header("Home Page")
        st.write("This is the home page of the app.")
        hommie()
    
    elif choice == "About":
        st.header("About Page")
        st.write("This page contains information about the app.")
        
    elif choice == "Contact":
        st.header("Contact Page")
        st.write("This page contains contact details.")
    
if __name__ == "__main__":
    main()
