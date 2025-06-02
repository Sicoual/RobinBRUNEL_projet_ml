# pages/entrainement.py
import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

def charger_feedback_si_dispo(df_original, target_col):
    """Recharge les feedbacks utilisateur comme données d'entraînement supplémentaires"""
    feedback_path = "models/feedback.csv"
    if os.path.exists(feedback_path):
        st.info("📥 Feedback utilisateur détecté – fusion avec les données d'origine")
        feedback = pd.read_csv(feedback_path)
        if target_col in feedback.columns:
            return pd.concat([df_original, feedback], ignore_index=True)
        else:
            st.warning("⚠️ Le fichier de feedback n'a pas la bonne colonne cible.")
    return df_original

def app():
    st.header("⚙️ Entraînement du modèle")

    df = pd.read_csv("data/vin.csv")

    # Option pour intégrer les feedbacks
    if st.checkbox("🔁 Inclure les corrections utilisateur (feedback.csv)", value=True):
        df = charger_feedback_si_dispo(df, target_col=st.selectbox("Cible temporaire", df.columns, index=len(df.columns)-1, label_visibility="collapsed"))

    target = st.selectbox("🎯 Choisissez la variable cible", df.columns)
    features = st.multiselect("🧮 Choisissez les variables explicatives", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target])

    if not features:
        st.warning("Veuillez sélectionner au moins une variable explicative.")
        return

    X = df[features]
    y = df[target]

    is_classification = y.dtype == "object" or y.nunique() <= 10
    le = None
    if is_classification:
        st.info("🧪 Problème de classification détecté")
        le = LabelEncoder()
        y = le.fit_transform(y)
        algos = {
            "Random Forest": RandomForestClassifier(),
            "Régression Logistique": LogisticRegression(max_iter=1000),
            "SVM": SVC(probability=True)
        }
    else:
        st.info("📈 Problème de régression détecté")
        algos = {
            "Random Forest": RandomForestRegressor(),
            "Régression Linéaire": LinearRegression(),
            "SVR": SVR()
        }

    algo_name = st.selectbox("⚙️ Choisissez un algorithme", list(algos.keys()))
    test_size = st.slider("📊 Taille du jeu de test (%)", 10, 50, 20) / 100
    model = algos[algo_name]

    if st.button("🚀 Entraîner le modèle"):
        X = pd.get_dummies(X)  # encodage si besoin
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)

        os.makedirs("models", exist_ok=True)
        with open("models/modele_entraine.pkl", "wb") as f:
            pickle.dump(pipe, f)
        if le:
            with open("models/label_encoder.pkl", "wb") as f:
                pickle.dump(le, f)

        # Ajout des éléments en session
        st.session_state['model'] = pipe
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['label_encoder'] = le
        st.session_state['algo_name'] = algo_name
        st.session_state['feature_names'] = list(X.columns)

        st.success(f"✅ Modèle **{algo_name}** entraîné avec succès et prêt à être évalué !")
