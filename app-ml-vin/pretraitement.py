import streamlit as st
import pandas as pd

def app():
    st.write("## Prétraitement des données")

    df = pd.read_csv("data/vin.csv")

    st.write("Nombre de valeurs manquantes par colonne :")
    st.write(df.isnull().sum())

    # Exemple simple d'imputation
    st.write("Imputation des valeurs manquantes (remplacer par la moyenne) :")
    df_filled = df.fillna(df.mean())
    st.write(df_filled.isnull().sum())
