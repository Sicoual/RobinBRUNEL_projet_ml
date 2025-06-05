import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lazypredict.Supervised import LazyClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

def save_dl_artifacts(model, scaler, encoder):
    folder = "models_dl"
    os.makedirs(folder, exist_ok=True)
    model.save(os.path.join(folder, "model_dl.h5"))
    joblib.dump(scaler, os.path.join(folder, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(folder, "encoder.pkl"))

def load_dl_artifacts():
    folder = "models_dl"
    model_path = os.path.join(folder, "model_dl.h5")
    scaler_path = os.path.join(folder, "scaler.pkl")
    encoder_path = os.path.join(folder, "encoder.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        return model, scaler, encoder
    return None, None, None

def app():
    st.header("🧪 Analyse avancée & Optimisation des modèles")

    # Charger les artefacts DL si présents
    model_dl, scaler_dl, encoder_dl = load_dl_artifacts()
    if model_dl and scaler_dl and encoder_dl:
        st.session_state["model"] = model_dl
        st.session_state["scaler"] = scaler_dl
        st.session_state["encoder"] = encoder_dl

    # 🔄 Choix de la source de données
    data_source = st.radio("Choisir les données à utiliser :", ["Données nettoyées", "Données brutes"])
    path_clean = "data/donnees_traitees.csv"
    path_raw = "data/vin.csv"

    if data_source == "Données nettoyées" and os.path.exists(path_clean):
        df = pd.read_csv(path_clean)
        st.success("✅ Données nettoyées chargées.")
    elif os.path.exists(path_raw):
        df = pd.read_csv(path_raw)
        st.info("📥 Données brutes chargées.")
    else:
        st.error("❌ Aucun fichier de données valide trouvé.")
        return

    if "target" not in df.columns:
        st.error("La colonne 'target' est requise dans les données.")
        return

    X = df.drop(columns=["target"])
    y = df["target"]

    # Exclure 'id' des caractéristiques si présent
    features = list(X.columns)
    if 'id' in features:
        features.remove('id')
    st.session_state["X_columns"] = features

    # ----------------------------
    # Paramètres globaux du split (sur la page)
    # ----------------------------
    st.markdown("### ⚙️ Paramètres du dataset")
    test_size = st.slider("📊 Taille du jeu de test (%)", min_value=10, max_value=50, value=30, step=5) / 100
    random_state = st.number_input("🎲 Random State (aléatoire)", min_value=0, max_value=9999, value=42, step=1)

    # ----------------------------
    # LazyPredict et GridSearch dans la sidebar
    # ----------------------------
    with st.sidebar:
        st.subheader("⚡ Comparaison rapide avec LazyPredict")
        if st.button("Exécuter LazyPredict"):
            with st.spinner("⏳ En cours d'exécution..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X[features], y, test_size=test_size, stratify=y, random_state=random_state
                )
                clf = LazyClassifier(verbose=0, ignore_warnings=True)
                models, _ = clf.fit(X_train, X_test, y_train, y_test)
                st.dataframe(models)

        st.subheader("🔧 Optimisation par GridSearchCV")
        algo = st.selectbox("Choisissez un modèle à optimiser", ["Random Forest", "Logistic Regression", "SVM"])

        if algo == "Random Forest":
            model = RandomForestClassifier(random_state=random_state)
            param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
        elif algo == "Logistic Regression":
            model = LogisticRegression(max_iter=2000, random_state=random_state)
            param_grid = {"C": [0.1, 1.0, 10], "penalty": ["l2"]}
        else:  # SVM
            model = SVC(random_state=random_state, probability=True)
            param_grid = {"C": [0.1, 1.0, 10], "kernel": ["linear", "rbf"]}

        if st.button("Lancer GridSearch"):
            with st.spinner("🔍 Recherche des meilleurs hyperparamètres..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X[features], y, test_size=test_size, stratify=y, random_state=random_state
                )
                grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
                grid.fit(X_train, y_train)
                st.success("✅ Optimisation terminée")
                st.write("🔧 Meilleurs paramètres :", grid.best_params_)
                st.write(f"📊 Score sur test : {grid.score(X_test, y_test):.2%}")

    # ----------------------------
    # 3. Deep Learning avec Keras (sur la page)
    # ----------------------------
    st.subheader("🤖 Entraînement d'un modèle Deep Learning (Keras)")

    if st.button("Lancer le modèle Deep Learning"):
        with st.spinner("⚙️ Préparation des données et du modèle..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X[features], y, test_size=test_size, stratify=y, random_state=random_state
            )

            # Standardisation
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Encodage cible
            encoder = LabelEncoder()
            y_train_enc = encoder.fit_transform(y_train)
            y_test_enc = encoder.transform(y_test)
            num_classes = len(np.unique(y_train_enc))

            # One-hot si plusieurs classes
            y_train_cat = to_categorical(y_train_enc)
            y_test_cat = to_categorical(y_test_enc)

            # Construction du modèle
            model_dl = Sequential([
                Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(num_classes, activation='softmax')
            ])

            model_dl.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            early_stop = EarlyStopping(patience=5, restore_best_weights=True)

            # Entraînement
            history = model_dl.fit(
                X_train_scaled, y_train_cat,
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )

            # Évaluation
            loss, accuracy = model_dl.evaluate(X_test_scaled, y_test_cat, verbose=0)
            st.success(f"✅ Précision sur le jeu de test : {accuracy:.2%}")

            # Rapport
            y_pred = np.argmax(model_dl.predict(X_test_scaled), axis=1)
            report = classification_report(y_test_enc, y_pred, target_names=encoder.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).T)

            # Courbes d’apprentissage
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(history.history["loss"], label="Train Loss")
            ax.plot(history.history["val_loss"], label="Validation Loss")
            ax.set_title("Évolution de la perte")
            ax.set_xlabel("Époques")
            ax.set_ylabel("Perte")
            ax.legend()
            st.pyplot(fig)

            # Sauvegarder pour prédiction future
            save_dl_artifacts(model_dl, scaler, encoder)

            # Mettre en session_state
            st.session_state["model"] = model_dl
            st.session_state["scaler"] = scaler
            st.session_state["encoder"] = encoder

    # ----------------------------
    # Prédiction nouveau vin (sur la page)
    # ----------------------------
    st.subheader("🔮 Prédire un nouveau vin")

    if "scaler" in st.session_state and "model" in st.session_state and "encoder" in st.session_state and "X_columns" in st.session_state:
        with st.form("form_new_wine"):
            st.write("Entrez les caractéristiques du vin à prédire (laissez vide pour utiliser la valeur médiane):")
            input_data = {}
            for col in st.session_state["X_columns"]:
                val = st.text_input(f"{col}", value="", key=f"input_{col}")
                if val.strip() == "":
                    input_data[col] = np.nan  # valeur manquante
                else:
                    try:
                        input_data[col] = float(val)
                    except ValueError:
                        st.error(f"Valeur invalide pour {col}.")
                        return

            submitted = st.form_submit_button("Prédire")

        if submitted:
            X_new = pd.DataFrame([input_data])

            # Réindexer les colonnes dans le bon ordre et sans 'id'
            X_new = X_new.reindex(columns=st.session_state["X_columns"])

            # Remplacer NaN par la médiane des données d'entraînement
            for col in st.session_state["X_columns"]:
                if X_new[col].isna().any():
                    median_val = X[col].median()
                    X_new[col].fillna(median_val, inplace=True)

            # Standardisation
            X_new_scaled = st.session_state["scaler"].transform(X_new)

            # Prédiction
            proba = st.session_state["model"].predict(X_new_scaled)
            pred_class_idx = np.argmax(proba, axis=1)[0]
            pred_class = st.session_state["encoder"].classes_[pred_class_idx]
            confidence = proba[0][pred_class_idx]

            st.write(f"### 🎯 Prédiction : **{pred_class}**")
            st.write(f"💡 Niveau de confiance : **{confidence:.2%}**")
    else:
        st.warning("⚠️ Entraîne d'abord le modèle Deep Learning avant de prédire un nouveau vin.")

if __name__ == "__main__":
    app()
