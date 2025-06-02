import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    st.write("## Évaluation du modèle")

    df = pd.read_csv("data/vin.csv")

    if 'quality' not in df.columns:
        st.error("La colonne 'quality' n'existe pas dans le dataset.")
        return

    X = df.drop('quality', axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.write("Rapport de classification :")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.json(report)

    st.write("Matrice de confusion :")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
