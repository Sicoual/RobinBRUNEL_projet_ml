# exploration.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    st.header("🔍 Exploration des données")

    try:
        df = pd.read_csv("data/vin.csv")
    except FileNotFoundError:
        st.error("Le fichier 'data/vin.csv' est introuvable.")
        return

    st.subheader("Aperçu du jeu de données")
    st.dataframe(df.head())

    st.subheader("Statistiques descriptives")
    st.write(df.describe())

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    selected_cols = st.multiselect("Colonnes numériques à visualiser :", numeric_cols, default=numeric_cols[:3])

    for col in selected_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution de {col}")
        st.pyplot(fig)

    if st.checkbox("Afficher le pairplot"):
        if len(selected_cols) > 1:
            pairplot_fig = sns.pairplot(df[selected_cols])
            st.pyplot(pairplot_fig)
        else:
            st.warning("Veuillez sélectionner au moins deux colonnes pour afficher un pairplot.")

    st.subheader("Matrice de corrélation")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)