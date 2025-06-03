import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    st.header("🔍 Exploration des données")
    df = pd.read_csv("data/vin.csv")

    st.write("### Aperçu des données")
    st.dataframe(df.head())

    st.write("### Statistiques descriptives")
    st.write(df.describe())

    st.write("### Sélection de colonnes pour distribution")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    selected_cols = st.multiselect(
        "Choisissez une ou plusieurs colonnes numériques", 
        numeric_cols, 
        default=numeric_cols[:3]
    )
    
    for col in selected_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    if st.checkbox("Afficher pairplot"):
        # On affiche le pairplot uniquement sur les colonnes numériques sélectionnées
        if len(selected_cols) > 1:
            pairplot_fig = sns.pairplot(df[selected_cols])
            st.pyplot(pairplot_fig.fig)
        else:
            st.warning("Sélectionnez au moins 2 colonnes pour afficher un pairplot.")

    st.write("### Corrélation (colonnes numériques uniquement)")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
