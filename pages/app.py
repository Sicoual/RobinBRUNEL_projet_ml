import streamlit as st
import exploration
import traitement
import entrainement
import deep_learning
import evaluation

# Configuration de la page
st.set_page_config(page_title="🍇 Application Machine Learning - Vin", layout="wide")

# Titre général
st.title("🍇 Application Machine Learning - Vin")

# Sidebar pour la navigation
st.sidebar.markdown("## 📚 Navigation")
page = st.sidebar.radio("Choisissez une page :", [
    "Accueil",
    "1.Exploration",
    "2.Traitement",
    "3.Entrainement",
    "4.Deep Learning",
    "5.Évaluation"
])

# Affichage selon la page choisie
if page == "Accueil":
    st.subheader("Bienvenue sur l'application d'analyse de vin 🍇")
    st.markdown("""
    Cette application vous permet d'explorer un pipeline complet de Machine Learning :
    - 1.Exploration des données
    - 2.Traitement
    - 3.Entrainement
    - 4.Deep Learning
    - 5.Évaluation
    """)
elif page == "1.Exploration":
    exploration.app()
elif page == "2.Traitement":
    traitement.app()
elif page == "3.Entrainement":
    entrainement.app()
elif page == "4.Deep Learning":
    deep_learning.app()
elif page == "5.Évaluation":
    evaluation.app()
else:
    st.error("Page inconnue. Veuillez sélectionner une page valide depuis la sidebar.")
