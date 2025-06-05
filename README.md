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
├── data/
│   └── vin.csv                    # Jeu de données CSV (à fournir manuellement)

├── models/                       # Modèles ML sauvegardés (joblib / Keras)
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── label_encoder.joblib
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── model_dl.h5               # Modèle Deep Learning Keras
│   └── feature_names.txt

├── models_dl/
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── model_dl.h5               # Modèle Deep Learning Keras

├── pages/
│   ├── exploration.py             # Exploration interactive des données
│   ├── pretraitement.py           # Prétraitement avancé (imputation multiple, suppression)
│   ├── training.py                # Entraînement ML & DL, sauvegarde des artefacts
│   ├── evaluation.py              # Évaluation, prédiction avec gestion des valeurs manquantes
│   ├── deep_learning.py           # Module dédié au Deep Learning (Keras)
│   └── app.py                    # Point d’entrée principal (navigation Streamlit)

├── setup.py                      # Script d’installation des dépendances
├── run.py                        # Script de lancement de l’application Streamlit
├── requirements.txt              # Liste des packages requis
├── README.md                     # Documentation utilisateur / notes projet
└── .gitignore                    # Fichiers/dossiers ignorés par Git
```

## Contributeurs
- Robin BRUNEL

## 🚀 Lancer l'application

 **Clonez le repository** :

    ```bash
    git clone https://github.com/Sicoual/RobinBRUNEL_projet_ml
    cd 
    ```

2. **Allez dans le dossier src** :

    ```bash
    cd src

### 🔧 Installation et lancement
Exécute ce script avec Python cela installeras toute les dépendance et l'environnement virtuel  (un seule fois) :
```bash
python setup.py
```
### Exécute ce script avec Python lanceras l'environnement virtuel et l'aplication steamlit
```bash
python run.py
```