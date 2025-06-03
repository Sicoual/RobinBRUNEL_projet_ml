import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os
import numpy as np
import joblib

def plot_confusion_matrix(cm, classes, title='Matrice de confusion'):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Réel')
    ax.set_title(title)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes, rotation=0)
    st.pyplot(fig)

def plot_roc_curve(y_test_bin, y_score, n_classes, class_labels, title='Courbe ROC'):
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax.plot(fpr[i], tpr[i], lw=2,
                label=f"{class_labels[i]} (AUC = {roc_auc[i]:.2f})")
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

    data_source = st.radio("Choisir les données à utiliser :", ["Données nettoyées", "Données brutes"])

    path_clean = "data/donnees_traitees.csv"
    path_raw = "data/vin.csv"

    if data_source == "Données nettoyées":
        if os.path.exists(path_clean):
            df = pd.read_csv(path_clean)
            st.success("✅ Données nettoyées chargées.")
        else:
            st.warning("⚠️ Données nettoyées non trouvées. Utilisation des données brutes.")
            if os.path.exists(path_raw):
                df = pd.read_csv(path_raw)
                st.info("📥 Données brutes chargées à la place.")
            else:
                st.error("❌ Aucun fichier de données trouvé.")
                return
    else:
        if os.path.exists(path_raw):
            df = pd.read_csv(path_raw)
            st.success("✅ Données brutes chargées.")
        else:
            st.error("❌ Fichier de données brutes non trouvé.")
            return

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
    else:
        y_enc = y
        le = None

    class_names = ["Vin équilibré", "Vin amer", "Vin sucré"]
    label_dict = {0: "Vin équilibré", 1: "Vin amer", 2: "Vin sucré"}

    # --- Paramètres split modifiables par l'utilisateur ---
    test_size = st.slider("Proportion des données pour le test (test_size)", 0.1, 0.9, 0.5, 0.05)
    random_state = st.number_input("random_state (graine aléatoire)", min_value=0, max_value=1000, value=50, step=1)

    # Split automatique à chaque changement de paramètre
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc)

    st.write(
        f"Données divisées en train ({len(X_train)}) / test ({len(X_test)})\n"
        f"- test_size = {test_size} (proportion des données affectées au test)\n"
        f"- random_state = {random_state} (graine aléatoire pour reproductibilité)"
    )

    model_options = {
        "Random Forest": make_pipeline(SimpleImputer(strategy='mean'), RandomForestClassifier(random_state=42)),
        "Régression Logistique": make_pipeline(SimpleImputer(strategy='mean'), LogisticRegression(max_iter=2000)),
        "SVM": make_pipeline(SimpleImputer(strategy='mean'), SVC(probability=True))
    }

    selected_models = st.multiselect(
        "Modèles à entraîner", list(model_options.keys()), default=list(model_options.keys()))

    if st.button("Lancer l'entraînement"):
        results = {}
        feature_names = list(X.columns)
        n_classes = len(class_names)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))

        for name in selected_models:
            model = model_options[name]
            model.fit(X_train, y_train)

            X_test_reindexed = X_test.reindex(columns=feature_names, fill_value=0)
            y_pred = model.predict(X_test_reindexed)

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            try:
                if hasattr(model.named_steps[model.steps[-1][0]], "predict_proba"):
                    y_score = model.predict_proba(X_test_reindexed)
                elif hasattr(model.named_steps[model.steps[-1][0]], "decision_function"):
                    decision_scores = model.decision_function(X_test_reindexed)
                    if n_classes == 2 and len(decision_scores.shape) == 1:
                        y_score = np.vstack([1 - decision_scores, decision_scores]).T
                    else:
                        y_score = decision_scores
                else:
                    y_score = None
            except Exception:
                y_score = None

            results[name] = {
                "model": model,
                "report": report,
                "confusion_matrix": cm,
                "y_pred": y_pred,
                "y_score": y_score
            }

            st.success(f"Modèle {name} entraîné.")

            st.write(f"### Rapport classification - {name}")
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report.style.apply(
                lambda row: ['font-weight: bold' if row.name in ['accuracy', 'macro avg', 'weighted avg'] else '' for _ in row],
                axis=1))
            
            plot_confusion_matrix(cm, classes=class_names, title=f"Matrice de confusion - {name}")

            if y_score is not None:
                plot_roc_curve(y_test_bin, y_score, n_classes, class_names, title=f"Courbe ROC - {name}")

            st.write("🔍 Exemple de prédictions :", [label_dict[int(p)] for p in y_pred[:5]])

        # Sauvegarde
        os.makedirs("models", exist_ok=True)
        for name, res in results.items():
            joblib.dump(res['model'], f"models/{name}.joblib")
        if le is not None:
            joblib.dump(le, "models/label_encoder.joblib")
        with open("models/feature_names.txt", "w") as f:
            for feat in feature_names:
                f.write(feat + "\n")

        st.success("✅ Entraînement terminé. Modèles sauvegardés dans le dossier models.")

if __name__ == "__main__":
    app()
