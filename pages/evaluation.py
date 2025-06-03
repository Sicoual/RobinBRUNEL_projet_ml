import streamlit as st
import pandas as pd
import numpy as np
import os

def app():
    st.header("🍷 Évaluation de nouveaux vins")

    if 'models' not in st.session_state or 'feature_names' not in st.session_state:
        st.warning("Aucun modèle entraîné. Veuillez d'abord entraîner un modèle.")
        return

    if "manual_entries" not in st.session_state:
        st.session_state.manual_entries = []

    st.subheader("Entrée des caractéristiques du vin")

    mode = st.radio("Mode de saisie :", ["Manuel", "CSV"])

    if mode == "Manuel":
        input_data = {}
        for feat in st.session_state['feature_names']:
            val = st.number_input(feat, value=0.0, key=feat)
            input_data[feat] = val

        if st.button("Ajouter ce vin"):
            st.session_state.manual_entries.append(input_data.copy())
            st.success("Vin ajouté. Vous pouvez en ajouter un autre.")

        if st.session_state.manual_entries:
            df_manual = pd.DataFrame(st.session_state.manual_entries)
            st.write("Vins ajoutés :", df_manual)
            X_input = df_manual
        else:
            X_input = pd.DataFrame()

    else:
        uploaded_file = st.file_uploader("Téléversez un fichier CSV", type=["csv"])
        if uploaded_file:
            df_uploaded = pd.read_csv(uploaded_file)
            st.write("Aperçu :", df_uploaded.head())
            # Ici on peut rajouter imputation plus avancée si besoin
            X_input = df_uploaded.reindex(columns=st.session_state['feature_names'], fill_value=0)
        else:
            X_input = pd.DataFrame()

    if not X_input.empty:
        st.subheader("Prédictions")

        for model_name in st.session_state['algo_names']:
            model = st.session_state['models'][model_name]
            y_pred = model.predict(X_input)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)
                df_proba = pd.DataFrame(proba, columns=st.session_state['class_names'])
                df_result = X_input.copy()
                df_result["Prédiction"] = y_pred
                df_result = pd.concat([df_result, df_proba], axis=1)
                st.dataframe(df_result.style.format("{:.2f}"))
            else:
                df_result = X_input.copy()
                df_result["Prédiction"] = y_pred
                st.dataframe(df_result)

        if st.checkbox("Ajouter ces vins au dataset d'entraînement ?"):
            # Inverse label encoder si besoin
            if 'label_encoder' in st.session_state:
                y_labels = model.predict(X_input)
                label_inv = st.session_state['label_encoder'].inverse_transform(y_labels)
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
            st.success("Données ajoutées à vin.csv")

            if mode == "Manuel":
                st.session_state.manual_entries = []
