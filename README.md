# 🍷 Application Machine Learning – Streamlit

## 🎯 Objectif
Créer une application web interactive illustrant un pipeline complet de machine learning sur le dataset `vin.csv`.

## 📦 Technologies
- Python
- Pandas, Scikit-learn, Seaborn, Matplotlib
- Streamlit

## Arborescence complète du projet
```plaintext
RobinBRUNEL_projet_ml/
│
├── data/
│   └── vin.csv                    # Fichier de données à ajouter manuellement
│
├── pages/
│   ├── Exploration.py           # Analyse exploratoire
│   ├── Prétraitement.py         # Nettoyage des données
│   ├── Modélisation.py          # Modélisation ML
│   ├── Évaluation.py            # Évaluation du modèle
│   └── app-ml-vin.py            # Point d’entrée Streamlit
├── setup.py                       # Script d’installation (exécution une seule fois)
├── run.py                         # Script de lancement de l'aplication (exécution à chaque lancement aprés l'installation)
├── requirements.txt               # Dépendances du projet
├── README.md                      # Documentation utilisateur
└── .gitignore                     # Exclusion Git
```

## Contributeurs
- Robin BRUNEL

## 🚀 Lancer l'application

### 🔧 1. Installation 
Exécute ce script avec Python cela installeras toute les dépendance et l'environnement virtuel  (un seule fois) :
```bash
python setup.py

### 🔧 2. Lancement  de steamlit
streamlit run app-ml-vin/app.py


### 