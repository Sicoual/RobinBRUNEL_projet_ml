import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def traiter_donnees(
    path_in="data/vin.csv",
    path_out="data/donnees_traitees.csv",
    num_strategy="mean",
    cat_strategy="most_frequent",
    fill_value="Inconnu",
    drop_cols=None,
    standardize_cols=None,
    verbose=True
):
    # Chargement des données
    df = pd.read_csv(path_in)
    if verbose:
        print(f"Chargement des données depuis {path_in} - shape: {df.shape}")

    # Colonnes numériques et catégoriques
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if verbose:
        print(f"Colonnes numériques détectées : {numeric_cols}")
        print(f"Colonnes catégoriques détectées : {categorical_cols}")

    # Imputation numérique
    if num_strategy and numeric_cols:
        imputer_num = SimpleImputer(strategy=num_strategy)
        df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
        if verbose:
            print(f"Imputation numérique réalisée avec la stratégie '{num_strategy}'")

    # Imputation catégorique
    if cat_strategy and categorical_cols:
        if cat_strategy == "constant":
            imputer_cat = SimpleImputer(strategy="constant", fill_value=fill_value)
            if verbose:
                print(f"Imputation catégorique constante avec la valeur '{fill_value}'")
        else:
            imputer_cat = SimpleImputer(strategy=cat_strategy)
            if verbose:
                print(f"Imputation catégorique réalisée avec la stratégie '{cat_strategy}'")
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

    # Suppression de colonnes
    if drop_cols:
        # Ne supprimer que les colonnes présentes dans df
        cols_to_drop = [col for col in drop_cols if col in df.columns]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        if verbose:
            print(f"Colonnes supprimées : {cols_to_drop}")

    # Standardisation
    if standardize_cols:
        # Ne standardiser que les colonnes numériques présentes dans df
        cols_to_standardize = [col for col in standardize_cols if col in df.columns]
        if cols_to_standardize:
            scaler = StandardScaler()
            df[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])
            if verbose:
                print(f"Standardisation appliquée sur les colonnes : {cols_to_standardize}")
        else:
            if verbose:
                print("Aucune colonne valide trouvée pour la standardisation.")

    # Sauvegarde
    df.to_csv(path_out, index=False)
    if verbose:
        print(f"Données traitées enregistrées dans {path_out} - shape: {df.shape}")

    return df


def app():
    st.title("🛠️ Traitement des données")

    st.markdown("""
    Cette page permet de traiter les données brutes en effectuant :
    - Imputation des valeurs manquantes
    - Suppression de colonnes inutiles
    - Standardisation de certaines colonnes
    """)

    path_in = st.text_input("Chemin fichier données brutes", "data/vin.csv")
    path_out = st.text_input("Chemin fichier sauvegarde", "data/donnees_traitees.csv")

    # Charger le dataframe pour récupérer les colonnes (sans crash si fichier absent)
    try:
        df = pd.read_csv(path_in)
        all_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    except Exception as e:
        st.error(f"Erreur chargement fichier : {e}")
        all_columns = []
        numeric_cols = []
        categorical_cols = []

    num_strategy = st.selectbox("Stratégie imputation numérique", ["mean", "median", "most_frequent", None])
    cat_strategy = st.selectbox("Stratégie imputation catégorique", ["most_frequent", "constant", None])

    # Si imputation constante sélectionnée, choisir la colonne catégorique pour appliquer la valeur constante
    fill_value = None
    if cat_strategy == "constant":
        if categorical_cols:
            col_fill = st.selectbox("Colonne catégorique pour imputation constante", categorical_cols)
        else:
            col_fill = None
            st.warning("Aucune colonne catégorique disponible pour imputation constante")
        fill_value = st.text_input("Valeur pour imputation constante (catégorique)", "Inconnu")
    else:
        col_fill = None  # non utilisée si pas imputation constante

    drop_cols = st.multiselect("Colonnes à supprimer", options=all_columns)
    standardize_cols = st.multiselect("Colonnes à standardiser", options=numeric_cols)

    if st.button("Lancer le traitement"):
        # Si imputation constante, on ne traite que la colonne choisie pour la valeur constante
        # Mais ton code traite toutes les colonnes catégoriques, donc ici on laisse comme avant
        df_result = traiter_donnees(
            path_in=path_in,
            path_out=path_out,
            num_strategy=num_strategy if num_strategy != "None" else None,
            cat_strategy=cat_strategy if cat_strategy != "None" else None,
            fill_value=fill_value if cat_strategy == "constant" else "Inconnu",
            drop_cols=drop_cols if drop_cols else None,
            standardize_cols=standardize_cols if standardize_cols else None,
            verbose=False
        )
        st.success(f"Données traitées enregistrées dans `{path_out}`")
        st.write(df_result.head())
