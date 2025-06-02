import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

def impute_missing_columns(df, feature_names, imputers):
    """
    Impute missing columns in df using provided imputers dict {col: model}.
    If a column is missing and no imputer available, fill with 0 or mean fallback.
    """
    missing = set(feature_names) - set(df.columns)
    for col in missing:
        if col in imputers:
            # Imputation via modèle prédictif
            # On prédit la colonne manquante à partir des autres colonnes existantes
            features_for_imputation = [f for f in feature_names if f != col and f in df.columns]
            if len(features_for_imputation) == 0:
                # Pas d’info pour imputer, on remplit à 0
                df[col] = 0.0
            else:
                X_imp = df[features_for_imputation]
                # Remplir les NaN dans X_imp par 0 (ou moyenne si stockée)
                X_imp = X_imp.fillna(0)
                imputer = imputers[col]
                df[col] = imputer.predict(X_imp)
        else:
            # Pas d’imputer, on remplit avec la moyenne stockée sinon 0
            mean_value = st.session_state.get("feature_means", {}).get(col, 0.0)
            df[col] = mean_value
    return df

def app():
    st.header("🍷 Évaluation de nouveaux vins")

    if 'models' not in st.session_state or 'feature_names' not in st.session_state:
        st.warning("Aucun modèle entraîné. Veuillez d'abord entraîner un modèle.")
        return

    if "manual_entries" not in st.session_state:
        st.session_state.manual_entries = []

    st.subheader("🔢 Entrée des caractéristiques du vin")

    mode = st.radio("Choisir le mode de saisie :", ["📝 Manuel", "📁 CSV"])

    if mode == "📝 Manuel":
        input_data = {}
        for feature in st.session_state['feature_names']:
            val = st.number_input(f"{feature}", value=0.0, key=feature)
            input_data[feature] = val

        if st.button("➕ Ajouter ce vin à la liste"):
            st.session_state.manual_entries.append(input_data.copy())
            st.success("Vin ajouté. Vous pouvez en ajouter un autre.")

        if st.session_state.manual_entries:
            df_manual = pd.DataFrame(st.session_state.manual_entries)
            st.write("Vins ajoutés :", df_manual)
            X_input = df_manual
        else:
            X_input = pd.DataFrame()

    else:
        uploaded_file = st.file_uploader("📁 Téléversez un fichier CSV", type=["csv"])
        if uploaded_file:
            df_uploaded = pd.read_csv(uploaded_file)
            st.write("Aperçu des données importées :", df_uploaded.head())

            # Imputation intelligente des colonnes manquantes
            imputers = st.session_state.get("imputers", {})
            df_filled = impute_missing_columns(df_uploaded, st.session_state['feature_names'], imputers)
            
            X_input = df_filled[st.session_state['feature_names']]
        else:
            X_input = pd.DataFrame()

    if not X_input.empty:
        st.subheader("🤖 Prédictions")

        for model_name in st.session_state['algo_names']:
            model = st.session_state['models'][model_name]
            st.write(f"### Modèle : {model_name}")

            y_pred = model.predict(X_input)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)
                df_proba = pd.DataFrame(proba, columns=st.session_state.get('class_names', []))
                df_result = X_input.copy()
                df_result["Prédiction"] = y_pred
                df_result = pd.concat([df_result, df_proba], axis=1)
                st.dataframe(df_result.style.format("{:.2f}"))
            else:
                df_result = X_input.copy()
                df_result["Prédiction"] = y_pred
                st.dataframe(df_result)

        # Ajout au dataset vin.csv
        if st.checkbox("📥 Ajouter ces vins au dataset d'entraînement ?"):
            if "label_encoder" in st.session_state:
                y_labels = model.predict(X_input)
                label_inv = st.session_state["label_encoder"].inverse_transform(y_labels)
                X_input["target"] = label_inv
            else:
                X_input["target"] = y_pred

            csv_path = "data/vin.csv"
            if os.path.exists(csv_path):
                df_old = pd.read_csv(csv_path)
                final_df = pd.concat([df_old, X_input], ignore_index=True)
            else:
                final_df = X_input

            final_df.to_csv(csv_path, index=False)
            st.success("✅ Données ajoutées à vin.csv")

            if mode == "📝 Manuel":
                st.session_state.manual_entries = []
