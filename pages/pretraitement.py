import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer

def app():
    st.header("🧹 Prétraitement des données")
    df = pd.read_csv("data/vin.csv")

    st.write("Valeurs manquantes par colonne :")
    st.write(df.isnull().sum())

    # Correspondance entre options en français et valeurs acceptées par SimpleImputer
    mapping = {
        "Aucune": None,
        "Moyenne": "mean",
        "Médiane": "median",
        "Valeur la plus fréquente": "most_frequent"
    }

    method_fr = st.selectbox("Méthode d'imputation", list(mapping.keys()))
    strategy = mapping[method_fr]

    # Sélection des colonnes numériques uniquement
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if strategy is not None and numeric_cols:
        imputer = SimpleImputer(strategy=strategy)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    cols_to_drop = st.multiselect("Colonnes à supprimer", df.columns)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        st.success(f"Colonnes supprimées: {cols_to_drop}")

    st.write("Données après prétraitement :")
    st.dataframe(df.head())
