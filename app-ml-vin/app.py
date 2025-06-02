import streamlit as st
import exploration
import pretraitement
import modelisation
import evaluation

st.set_page_config(page_title="App ML Vin", layout="wide")

st.title("🍷 Application Machine Learning - Vin")

st.sidebar.markdown("## 📚 Navigation")
page = st.sidebar.radio("Choisissez une page :", [
    "🏠 Accueil",
    "1️⃣ Exploration",
    "2️⃣ Prétraitement",
    "3️⃣ Modélisation",
    "4️⃣ Évaluation"
])

if page == "🏠 Accueil":
    st.subheader("Bienvenue sur l'application d'analyse de vin 🍇")
    st.markdown("""
    Cette application vous permet d'explorer un pipeline complet de Machine Learning :
    - Exploration des données
    - Prétraitement
    - Modélisation
    - Évaluation
    """)
elif page == "1️⃣ Exploration":
    exploration.app()
elif page == "2️⃣ Prétraitement":
    pretraitement.app()
elif page == "3️⃣ Modélisation":
    modelisation.app()
elif page == "4️⃣ Évaluation":
    evaluation.app()
