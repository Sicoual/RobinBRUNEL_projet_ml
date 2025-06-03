import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def app():
    st.header("🧹 Nettoyage et Prétraitement des données")

    # Chargement des données
    df = pd.read_csv("data/vin.csv")

    st.markdown("### Aperçu initial des données")
    st.dataframe(df.head())

    # Gestion valeurs manquantes
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown("### Remplissage des valeurs manquantes")

    # Imputation numérique
    num_strategy = st.selectbox(
        "Méthode d'imputation (numérique) :",
        ["Ne rien faire", "Moyenne", "Médiane", "Valeur la plus fréquente"]
    )
    # Imputation catégorique
    cat_strategy = st.selectbox(
        "Méthode d'imputation (catégorique) :",
        ["Ne rien faire", "Valeur la plus fréquente", "Valeur personnalisée"]
    )
    constant_fill_value = None
    if cat_strategy == "Valeur personnalisée":
        constant_fill_value = st.text_input("Valeur personnalisée :", value="Inconnu")

    mapping = {
        "Ne rien faire": None,
        "Moyenne": "mean",
        "Médiane": "median",
        "Valeur la plus fréquente": "most_frequent",
        "Valeur personnalisée": "constant"
    }

    if mapping[num_strategy] and df[numeric_cols].isnull().sum().sum() > 0:
        imputer_num = SimpleImputer(strategy=mapping[num_strategy])
        df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
        st.success(f"✅ Imputation numérique : {num_strategy}")

    if mapping[cat_strategy] and df[categorical_cols].isnull().sum().sum() > 0:
        if cat_strategy == "Valeur personnalisée":
            imputer_cat = SimpleImputer(strategy="constant", fill_value=constant_fill_value)
        else:
            imputer_cat = SimpleImputer(strategy=mapping[cat_strategy])
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
        st.success(f"✅ Imputation catégorique : {cat_strategy}")

    st.markdown("### Suppression de colonnes")
    cols_to_drop = st.multiselect("Colonnes à supprimer :", df.columns)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        st.success(f"Colonnes supprimées : {', '.join(cols_to_drop)}")

    st.markdown("### Standardisation")
    cols_to_standardize = st.multiselect(
        "Colonnes numériques à standardiser :",
        [col for col in numeric_cols if col in df.columns]
    )
    if cols_to_standardize:
        scaler = StandardScaler()
        df[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])
        st.success("✅ Standardisation appliquée.")

    st.markdown("### Données prétraitées")
    st.dataframe(df.head())

    # Sauvegarde dans session_state
    st.session_state['cleaned_df'] = df
    st.session_state['feature_names'] = [col for col in df.columns if col != "target"]

    st.success("✅ Données prétraitées sauvegardées pour entraînement.")
