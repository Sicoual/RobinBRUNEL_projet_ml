import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

def load_models():
    if not os.path.exists("models"):
        return {}, None, []
    model_files = [f for f in os.listdir("models") if f.endswith(".joblib") and f != "label_encoder.joblib"]
    models = {}
    for mf in model_files:
        name = mf.replace(".joblib", "")
        models[name] = joblib.load(f"models/{mf}")
    label_encoder = None
    if os.path.exists("models/label_encoder.joblib"):
        label_encoder = joblib.load("models/label_encoder.joblib")

    feature_names = []
    if os.path.exists("models/feature_names.txt"):
        with open("models/feature_names.txt", "r") as f:
            feature_names = f.read().splitlines()

    return models, label_encoder, feature_names

def app():
    st.header("\U0001F377 Évaluation de nouveaux vins")

    models, label_encoder, feature_names = load_models()
    if not models:
        st.warning("Aucun modèle trouvé. Veuillez d'abord entraîner les modèles dans l'onglet Entraînement.")
        return
    if not feature_names:
        st.error("Fichier feature_names.txt manquant. Entraînez les modèles à nouveau.")
        return

    mode = st.radio("Mode de saisie :", ["Manuel", "CSV"])

    if mode == "Manuel":
        input_data = {}
        for feat in feature_names:
            val = st.text_input(f"{feat} (laisser vide si inconnu)", key=f"input_{feat}")
            try:
                input_data[feat] = float(val) if val.strip() != "" else np.nan
            except ValueError:
                st.error(f"Valeur invalide pour {feat}")
                return

        if st.button("Prédire"):
            df_input = pd.DataFrame([input_data])

            for model_name, model in models.items():
                try:
                    y_pred = model.predict(df_input)
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction avec {model_name}: {e}")
                    continue

                df_result = df_input.copy()
                df_result["Prédiction"] = y_pred

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(df_input)
                    df_proba = pd.DataFrame(proba, columns=label_encoder.classes_ if label_encoder else [str(i) for i in range(proba.shape[1])])
                    df_result = pd.concat([df_result, df_proba], axis=1)
                    df_result["Confiance (%)"] = (df_proba.max(axis=1) * 100).round(2)

                if label_encoder:
                    try:
                        df_result["Nom du vin prédit"] = label_encoder.inverse_transform(y_pred)
                    except Exception:
                        pass

                st.write(f"### Résultat - {model_name}")
                st.dataframe(df_result)

    else:  # CSV upload
        uploaded_file = st.file_uploader("Téléversez un fichier CSV", type=["csv"])
        if uploaded_file:
            df_uploaded = pd.read_csv(uploaded_file)
            st.write("Aperçu :", df_uploaded.head())
            X_input = df_uploaded.reindex(columns=feature_names, fill_value=np.nan)

            if st.button("Prédire les vins du fichier"):
                for model_name, model in models.items():
                    try:
                        y_pred = model.predict(X_input)
                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction avec {model_name}: {e}")
                        continue

                    df_result = X_input.copy()
                    df_result["Prédiction"] = y_pred

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_input)
                        df_proba = pd.DataFrame(proba, columns=label_encoder.classes_ if label_encoder else [str(i) for i in range(proba.shape[1])])
                        df_result = pd.concat([df_result, df_proba], axis=1)
                        df_result["Confiance (%)"] = (df_proba.max(axis=1) * 100).round(2)

                    if label_encoder:
                        try:
                            df_result["Nom du vin prédit"] = label_encoder.inverse_transform(y_pred)
                        except Exception:
                            pass

                    st.write(f"### Résultat - {model_name}")
                    st.dataframe(df_result)

if __name__ == "__main__":
    app()
