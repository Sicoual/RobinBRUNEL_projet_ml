import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

def save_report_locally(report_dict, filename="models/rapport_classification.json"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=4)

def save_cm_locally(cm_df, filename="models/matrice_confusion.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cm_df.to_csv(filename, index=True)

def app():
    st.header("📊 Évaluation du modèle")

    if 'model' not in st.session_state:
        st.warning("⚠️ Veuillez entraîner un modèle dans l'onglet Modélisation avant d’évaluer.")
        return

    model = st.session_state['model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    le = st.session_state.get('label_encoder', None)
    algo_name = st.session_state.get('algo_name', "Modèle")

    y_pred = model.predict(X_test)

    # Classification ou régression
    if le is not None:
        y_test_labels = le.inverse_transform(y_test)
        y_pred_labels = le.inverse_transform(y_pred)
    else:
        y_test_labels = y_test
        y_pred_labels = y_pred

    # Classification
    if le is not None:
        report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        st.subheader(f"Rapport de classification - {algo_name}")
        st.dataframe(df_report.style.apply(
            lambda row: ['font-weight: bold' if row.name in ['accuracy', 'macro avg', 'weighted avg'] else '' for _ in row],
            axis=1))

        # Sauvegarde locale
        save_report_locally(report)

        # Téléchargement
        report_json = json.dumps(report, indent=4, ensure_ascii=False)
        st.download_button(
            label="📥 Télécharger le rapport de classification (JSON)",
            data=report_json,
            file_name="rapport_classification.json",
            mime="application/json"
        )

        # Matrice de confusion
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

        st.subheader("Matrice de confusion")
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Prédictions")
        ax.set_ylabel("Vraies étiquettes")
        plt.tight_layout()
        st.pyplot(fig)

        # Sauvegarde locale matrice confusion
        save_cm_locally(cm_df)

        # Téléchargement matrice confusion CSV
        csv_data = cm_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Télécharger la matrice de confusion (CSV)",
            data=csv_data,
            file_name="matrice_confusion.csv",
            mime="text/csv"
        )

    # Régression
    else:
        st.subheader("Évaluation Régression")
        mse = mean_squared_error(y_test_labels, y_pred_labels)
        r2 = r2_score(y_test_labels, y_pred_labels)
        st.write(f"Erreur quadratique moyenne (MSE): {mse:.4f}")
        st.write(f"Coefficient de détermination (R²): {r2:.4f}")
    
    # ---------------------------------------------------------
    # 🔍 PRÉDICTION INTERACTIVE + FEEDBACK UTILISATEUR
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🍷 Tester le modèle avec vos propres données")

    if 'feature_names' in st.session_state:
        with st.form("formulaire_prediction_perso"):
            inputs = {}
            for col in st.session_state['feature_names']:
                default_val = float(X_test[col].mean()) if col in X_test.columns else 0.0
                inputs[col] = st.number_input(f"{col}", value=default_val, format="%.4f")
            valider = st.form_submit_button("Prédire")

        if valider:
            # Création DataFrame à prédire
            input_df = pd.DataFrame([inputs])
            input_df_encoded = input_df.copy()

            # Gestion des colonnes manquantes (encodage)
            if set(X_test.columns) != set(input_df_encoded.columns):
                input_df_encoded = input_df_encoded.reindex(columns=X_test.columns, fill_value=0)

            y_pred_interactif = model.predict(input_df_encoded)[0]

            if le:
                prediction_humaine = le.inverse_transform([y_pred_interactif])[0]
                st.success(f"✅ Le modèle prédit : **{prediction_humaine}**")
            else:
                prediction_humaine = y_pred_interactif
                st.success(f"✅ Prédiction numérique : **{prediction_humaine:.4f}**")

            # Probabilités si possible
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df_encoded)[0]
                df_proba = pd.DataFrame([proba], columns=le.classes_)
                st.write("🔢 Probabilités :")
                st.dataframe(df_proba.style.format("{:.2%}"))

            # Feedback utilisateur
            st.markdown("### 🤖 Est-ce correct selon vous ?")
            avis = st.radio("Votre avis :", ["Oui", "Non"], key="feedback_choice")

            correction = None
            if avis == "Non" and le:
                correction = st.selectbox("Quelle est la vraie classe ?", le.classes_)

            if st.button("Envoyer feedback"):
                feedback = {
                    **inputs,
                    "prediction": prediction_humaine,
                    "correct": avis,
                    "true_label": correction if correction else prediction_humaine
                }
                feedback_path = "models/feedback.csv"
                try:
                    df_old = pd.read_csv(feedback_path)
                    df_new = pd.concat([df_old, pd.DataFrame([feedback])], ignore_index=True)
                except FileNotFoundError:
                    df_new = pd.DataFrame([feedback])

                df_new.to_csv(feedback_path, index=False)
                st.success("💾 Feedback enregistré dans `models/feedback.csv`.")
