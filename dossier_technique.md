Dossier Technique – Application Machine Learning

1. 🎯 Présentation du projet
L’objectif de ce projet est de construire une application web interactive utilisant Streamlit pour illustrer un pipeline complet de Machine Learning, appliqué à un jeu de données de caractéristiques physico-chimiques de vins (vin.csv).

Ce projet vise à mettre en valeur :

Les compétences en Data Science

L'utilisation d'une interface interactive avec Streamlit

Le développement d’un pipeline ML complet (de l’exploration à l’évaluation

2. 🏗️ Architecture de l'application
L’application est structurée autour de 5 modules principaux, chacun correspondant à une page dans Streamlit.

✅ Architecture :
Streamlit multi-pages

Navigation par st.sidebar.radio

Application lancée via app.py qui centralise l’appel des modules

Données lues dans le dossier ./data/vin.csv

3. 🧩 Description des modules
Page	Fonctionnalités principales
Exploration	Affichage du dataframe, statistiques descriptives, distribution des variables, pairplot
Prétraitement	Nettoyage des données, gestion des valeurs manquantes, standardisation, sélection de colonnes
Modélisation	Sélection d’algorithmes ML, split, entraînement, prédiction sur données test ou nouvelles
Évaluation	Visualisation des métriques : accuracy, F1, confusion matrix, ROC/PR curves, classification report

4. 🧠 Modèles de Machine Learning
📚 Algorithmes utilisés :
Logistic Regression

Random Forest Classifier

K-Nearest Neighbors (KNN)

⚙️ Pipeline ML :
Séparation Train/Test (train_test_split)

Standardisation (StandardScaler)

Modélisation avec scikit-learn

Sauvegarde du modèle possible via joblib

🔮 Fonctionnalités :
Prédictions en temps réel

Possibilité de charger ses propres données

Pipeline automatisé avec interaction utilisateur

5. 👤 Interaction Utilisateur
L’interface permet à l’utilisateur :

De naviguer entre les étapes du pipeline via la sidebar

De visualiser et explorer les données

De sélectionner les variables à inclure dans l’entraînement

De choisir l’algorithme utilisé pour la modélisation

De lancer l’entraînement et la prédiction par bouton

D’analyser les résultats via des métriques et des graphiques