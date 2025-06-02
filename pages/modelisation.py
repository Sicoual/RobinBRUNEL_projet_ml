import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def app():
    st.header("🤖 Modélisation")

    df = pd.read_csv("data/vin.csv")
    target = st.selectbox("Choisissez la variable cible", df.columns)
    X = df.drop(columns=[target])
    y = df[target]

    # Encodage des colonnes catégoriques en variables numériques
    X = pd.get_dummies(X)

    test_size = st.slider("Taille du test (%)", 10, 50, 20) / 100
    algo_name = st.selectbox("Algorithme", ["Random Forest", "Logistic Regression", "SVM"])

    algos = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
    }
    clf = algos[algo_name]

    if st.button("Entraîner le modèle"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        clf.fit(X_train, y_train)
        st.session_state['model'] = clf
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.success(f"Modèle {algo_name} entraîné avec succès !")
