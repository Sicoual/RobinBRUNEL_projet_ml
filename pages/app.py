import streamlit as st
import exploration
import pretraitement
import modelisation
import evaluation

# Configuration de la page
st.set_page_config(page_title="🍷 Application Machine Learning - Vin", layout="wide")

# Titre général
st.title("🍷 Application Machine Learning - Vin")

# Sidebar pour la navigation
st.sidebar.markdown("## 📚 Navigation")
page = st.sidebar.radio("Choisissez une page :", [
    "🏠 Accueil",
    "Exploration",
    "Prétraitement",
    "Modélisation",
    "Évaluation"
])

# Affichage selon la page choisie
if page == "🏠 Accueil":
    st.subheader("Bienvenue sur l'application d'analyse de vin 🍇")
    st.markdown("""
    Cette application vous permet d'explorer un pipeline complet de Machine Learning :
    - Exploration des données
    - Prétraitement
    - Modélisation
    - Évaluation
    """)
elif page == "Exploration":
    exploration.app()
elif page == "Prétraitement":
    pretraitement.app()
elif page == "Modélisation":
    modelisation.app()
elif page == "Évaluation":
    evaluation.app()
else:
    st.error("Page inconnue. Veuillez sélectionner une page valide depuis la sidebar.")
