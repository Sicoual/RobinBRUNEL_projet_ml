import streamlit as st 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def app():
    st.header("🧹 Nettoyage et Prétraitement des données")

    # Chargement des données
    df = pd.read_csv("data/vin.csv")

    # Copie originale pour suivi
    original_df = df.copy()

    st.markdown("### 1. Aperçu initial des données")
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing == 0:
        st.info("✅ Aucune valeur manquante détectée dans le jeu de données.")
    else:
        st.warning("⚠️ Données manquantes détectées :")
        st.write(missing_values[missing_values > 0])

    # Séparation colonnes numériques et catégoriques
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown("### 2. Remplissage des valeurs manquantes")

    # Imputation numérique
    if numeric_cols:
        st.write(f"**Colonnes numériques :** {', '.join(numeric_cols)}")
        num_strategy = st.selectbox(
            "Méthode d'imputation des colonnes numériques :",
            ["Ne rien faire", "Moyenne", "Médiane", "Valeur la plus fréquente"]
        )
    else:
        num_strategy = "Ne rien faire"

    # Imputation catégorique
    if categorical_cols:
        st.write(f"**Colonnes catégoriques :** {', '.join(categorical_cols)}")
        cat_strategy = st.selectbox(
            "Méthode d'imputation des colonnes catégoriques :",
            ["Ne rien faire", "Valeur la plus fréquente", "Valeur personnalisée"]
        )
        constant_fill_value = None
        if cat_strategy == "Valeur personnalisée":
            constant_fill_value = st.text_input("Valeur personnalisée :", value="Inconnu")
    else:
        cat_strategy = "Ne rien faire"

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

    st.markdown("### 3. Suppression de colonnes")
    cols_to_drop = st.multiselect("Colonnes à supprimer :", df.columns)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        st.success(f"Colonnes supprimées : {', '.join(cols_to_drop)}")

    st.markdown("### 4. Standardisation")
    cols_to_standardize = st.multiselect(
        "Colonnes numériques à standardiser :",
        [col for col in numeric_cols if col in df.columns]
    )

    if cols_to_standardize:
        scaler = StandardScaler()
        df_std = df.copy()
        df_std[cols_to_standardize] = scaler.fit_transform(df_std[cols_to_standardize])

        st.success("✅ Standardisation appliquée.")
        
        st.markdown("#### 🔄 Colonnes standardisées")
        st.dataframe(df_std[cols_to_standardize].head())

        st.markdown("#### 📊 Colonnes non standardisées")
        other_cols = [col for col in df.columns if col not in cols_to_standardize]
        st.dataframe(df_std[other_cols].head())

        df = df_std  # Mise à jour du DataFrame principal
    else:
        st.info("Aucune colonne sélectionnée pour la standardisation.")

    st.markdown("### 5. Vérification finale des valeurs manquantes")
    final_missing = df.isnull().sum()
    total_final_missing = final_missing.sum()
    if total_final_missing == 0:
        st.success("✅ Aucune valeur manquante après nettoyage.")
    else:
        st.warning("⚠️ Valeurs manquantes restantes :")
        st.write(final_missing[final_missing > 0])

    st.markdown("### 6. Données finales après nettoyage et standardisation")
    st.dataframe(df.head())
