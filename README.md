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
│   ├── exploration.py           # Analyse exploratoire
│   ├── pretraitement.py         # Nettoyage des données
│   ├── modélisation.py          # Modélisation ML
│   ├── evaluation.py            # Évaluation du modèle
│   └── app.py                   # Point d’entrée Streamlit
├── setup.py                       # Script d’installation (exécution une seule fois)
├── run.py                         # Script de lacement de l'aplication 
├── requirements.txt               # Dépendances du projet
├── README.md                      # Documentation utilisateur
└── .gitignore                     # Exclusion Git
```

## Contributeurs
- Robin BRUNEL

## 🚀 Lancer l'application

 **Clonez le repository** :

    ```bash
    git clone https://github.com/
    cd 
    ```

2. **Allez dans le dossier src** :

    ```bash
    cd src

### 🔧 1. Installation 
Exécute ce script avec Python cela installeras toute les dépendance et l'environnement virtuel  (un seule fois) :
```bash
python setup.py
```
### Exécute ce script avec Python lanceras l'environnement virtuel et l'aplication steamlit
```bash
python run.py
```
### 🔧 2. Lancement  de steamlit
```bash
streamlit run app-ml-vin/app.py
```
### 