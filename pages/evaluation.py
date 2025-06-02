import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    st.header("📊 Évaluation du modèle")

    if 'model' not in st.session_state:
        st.warning("Veuillez entraîner un modèle dans l'onglet Modélisation d'abord.")
        return

    model = st.session_state['model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']

    y_pred = model.predict(X_test)

    st.subheader("Rapport de classification")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.json(report)

    st.subheader("Matrice de confusion")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
