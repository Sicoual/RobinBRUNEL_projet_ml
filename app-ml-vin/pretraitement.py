import streamlit as st
import pandas as pd

def app():
    st.write("## Prétraitement des données")

    df = pd.read_csv("data/vin.csv")

    st.write("Nombre de valeurs manquantes par colonne :")
    st.write(df.isnull().sum())

    st.write("Imputation des valeurs manquantes :")
    df_filled = df.copy()

    # Imputation pour les colonnes numériques (moyenne)
    num_cols = df.select_dtypes(include='number').columns
    df_filled[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # Imputation pour les colonnes non numériques (mode)
    cat_cols = df.select_dtypes(exclude='number').columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_value = df[col].mode().dropna()
            if not mode_value.empty:
                df_filled[col] = df[col].fillna(mode_value[0])
            else:
                # Si pas de mode (colonne vide), remplir par une chaîne vide ou "inconnu"
                df_filled[col] = df[col].fillna("inconnu")

    st.write("Valeurs manquantes après imputation :")
    st.write(df_filled.isnull().sum())
