import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os
import numpy as np

def plot_confusion_matrix(cm, classes, title='Matrice de confusion'):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vérité terrain')
    ax.set_title(title)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes, rotation=0)
    st.pyplot(fig)

def plot_roc_curve(y_test_bin, y_score, n_classes, title='Courbe ROC'):
    # y_test_bin : données binarisées (one-hot) des labels test
    # y_score : scores de probabilité pour chaque classe
    fig, ax = plt.subplots(figsize=(6,5))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax.plot(fpr[i], tpr[i], lw=2,
                label=f"Classe {i} (AUC = {roc_auc[i]:.2f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux de faux positifs')
    ax.set_ylabel('Taux de vrais positifs')
    ax.set_title(title)
    ax.legend(loc="lower right")
    st.pyplot(fig)

def app():
    st.header("📂 Chargement et entraînement des modèles")

    csv_path = "data/vin.csv"
    if not os.path.exists(csv_path):
        st.error(f"Fichier {csv_path} non trouvé. Veuillez vérifier le chemin.")
        return

    df = pd.read_csv(csv_path)
    st.write("Aperçu des données :", df.head())

    if "target" not in df.columns:
        st.error("La colonne 'target' est obligatoire dans le fichier.")
        return

    X = df.drop(columns=["target"])
    y = df["target"]

    # Encodage des labels cibles
    if y.dtype == 'object':
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        st.session_state['label_encoder'] = le
        class_names = le.classes_
    else:
        y_enc = y
        class_names = np.unique(y_enc)
        le = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc)
    st.write(f"Données divisées en train ({len(X_train)}) / test ({len(X_test)})")

    model_options = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Régression Logistique": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True)
    }
    selected_models = st.multiselect(
        "Modèles à entraîner", list(model_options.keys()), default=list(model_options.keys()))

    if st.button("Lancer l'entraînement"):
        results = {}
        feature_names = list(X.columns)  # sauvegarde l’ordre des colonnes

        # Préparer labels binarisés pour ROC multi-classes
        n_classes = len(class_names)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))

        for name in selected_models:
            model = model_options[name]
            model.fit(X_train, y_train)

            # Réindexer X_test pour correspondre exactement aux colonnes d'entraînement
            X_test_reindexed = X_test.reindex(columns=feature_names, fill_value=0)
            y_pred = model.predict(X_test_reindexed)

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            # Calcul des scores de probas si possible pour la ROC
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test_reindexed)
            elif hasattr(model, "decision_function"):
                decision_scores = model.decision_function(X_test_reindexed)
                # Pour les modèles à 2 classes decision_function renvoie 1D, on reshape
                if n_classes == 2 and len(decision_scores.shape) == 1:
                    y_score = np.vstack([1 - decision_scores, decision_scores]).T
                else:
                    y_score = decision_scores
            else:
                y_score = None

            results[name] = {
                "model": model,
                "report": report,
                "confusion_matrix": cm,
                "y_pred": y_pred,
                "y_score": y_score
            }
            st.success(f"Modèle {name} entraîné.")

        # Stockage dans session_state
        st.session_state['models'] = {k: v['model'] for k, v in results.items()}
        st.session_state['reports'] = {k: v['report'] for k, v in results.items()}
        st.session_state['cms'] = {k: v['confusion_matrix'] for k, v in results.items()}
        st.session_state['y_pred'] = {k: v['y_pred'] for k, v in results.items()}
        st.session_state['y_score'] = {k: v['y_score'] for k, v in results.items()}
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['feature_names'] = feature_names
        st.session_state['algo_names'] = selected_models
        st.session_state['class_names'] = class_names

        st.success("✅ Entraînement terminé. Passez à l'onglet Évaluation pour voir les résultats.")

    # Affichage des métriques et graphiques si entraînement réalisé
    if 'reports' in st.session_state:
        st.subheader("Rapports disponibles")
        for algo_name in st.session_state['algo_names']:
            st.write(f"### Rapport pour {algo_name}")
            report = st.session_state['reports'][algo_name]
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report.style.apply(
                lambda row: ['font-weight: bold' if row.name in ['accuracy', 'macro avg', 'weighted avg'] else '' for _ in row], 
                axis=1))

            # Matrice de confusion graphique
            cm = st.session_state['cms'][algo_name]
            class_names = st.session_state['class_names']
            plot_confusion_matrix(cm, classes=class_names, title=f"Matrice de confusion - {algo_name}")

            # Courbe ROC si dispo
            y_score = st.session_state['y_score'][algo_name]
            if y_score is not None:
                y_test = st.session_state['y_test']
                n_classes = len(class_names)
                y_test_bin = label_binarize(y_test, classes=range(n_classes))
                plot_roc_curve(y_test_bin, y_score, n_classes, title=f"Courbe ROC - {algo_name}")
            else:
                st.info(f"Pas de courbe ROC disponible pour {algo_name}")

