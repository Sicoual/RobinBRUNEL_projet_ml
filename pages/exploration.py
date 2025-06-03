import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def app():
    st.header("🔍 Exploration des données")

    try:
        df = pd.read_csv("data/vin.csv")
    except FileNotFoundError:
        st.error("❌ Le fichier 'data/vin.csv' est introuvable.")
        return

    st.subheader("Aperçu du jeu de données")
    st.dataframe(df.head())

    st.subheader("Valeurs manquantes")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.write("🔎 Colonnes avec valeurs manquantes :")
        st.dataframe(missing)
        if st.checkbox("Remplacer les valeurs manquantes par la moyenne (colonnes numériques)"):
            df.fillna(df.mean(numeric_only=True), inplace=True)
            st.success("✅ Valeurs manquantes remplacées.")
    else:
        st.write("✅ Aucune valeur manquante détectée.")

    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe(include='all'))

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    st.subheader("Visualisation des variables numériques")
    selected_cols = st.multiselect("Colonnes numériques à visualiser :", numeric_cols, default=numeric_cols[:2])

    plot_type = st.selectbox("Type de graphique", ["Histogramme", "Boxplot", "Scatterplot (2 colonnes)"])

    for col in selected_cols:
        fig, ax = plt.subplots()
        if plot_type == "Histogramme":
            sns.histplot(df[col], kde=True, ax=ax)
        elif plot_type == "Boxplot":
            sns.boxplot(x=df[col], ax=ax)
        elif plot_type == "Scatterplot (2 colonnes)":
            if len(selected_cols) >= 2:
                sns.scatterplot(x=df[selected_cols[0]], y=df[selected_cols[1]], ax=ax)
                break
            else:
                st.warning("Veuillez sélectionner au moins 2 colonnes pour un scatterplot.")
                break
        ax.set_title(f"{plot_type} - {col}")
        st.pyplot(fig)

    if st.checkbox("Afficher le pairplot (⛔ lent si > 5 colonnes)"):
        if len(selected_cols) > 1:
            pairplot_fig = sns.pairplot(df[selected_cols])
            st.pyplot(pairplot_fig)
        else:
            st.warning("Sélectionnez au moins deux colonnes.")

    st.subheader("📊 Matrice de corrélation")
    if numeric_cols:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Aucune colonne numérique détectée.")

    if 'target' in df.columns:
        st.subheader("🎯 Répartition des classes (target)")
        fig, ax = plt.subplots()
        df['target'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Distribution de la target")
        st.pyplot(fig)
