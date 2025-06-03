import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lazypredict.Supervised import LazyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def app():
    st.header("🧪 Analyse avancée & Optimisation des modèles")

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

    # ----------------------------
    # 1. LazyPredict
    # ----------------------------
    st.subheader("⚡ Comparaison rapide avec LazyPredict")
    if st.button("Exécuter LazyPredict"):
        with st.spinner("⏳ En cours d'exécution..."):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
            clf = LazyClassifier(verbose=0, ignore_warnings=True)
            models, _ = clf.fit(X_train, X_test, y_train, y_test)
            st.dataframe(models)

    # ----------------------------
    # 2. GridSearchCV
    # ----------------------------
    st.subheader("🔧 Optimisation par GridSearchCV")
    algo = st.selectbox("Choisissez un modèle à optimiser", ["Random Forest", "Logistic Regression", "SVM"])

    if algo == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
    elif algo == "Logistic Regression":
        model = LogisticRegression(max_iter=2000)
        param_grid = {"C": [0.1, 1.0, 10], "penalty": ["l2"]}
    else:  # SVM
        model = SVC()
        param_grid = {"C": [0.1, 1.0, 10], "kernel": ["linear", "rbf"]}

    if st.button("Lancer GridSearch"):
        with st.spinner("🔍 Recherche des meilleurs hyperparamètres..."):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
            grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            st.success("✅ Optimisation terminée")
            st.write("🔧 Meilleurs paramètres :", grid.best_params_)
            st.write(f"📊 Score sur test : {grid.score(X_test, y_test):.2%}")
