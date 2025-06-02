import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def app():
    st.write("## Modélisation")

    df = pd.read_csv("data/vin.csv")

    # Supposons une target 'quality'
    if 'quality' not in df.columns:
        st.error("La colonne 'quality' n'existe pas dans le dataset.")
        return

    X = df.drop('quality', axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy du modèle RandomForest : {accuracy:.2f}")
