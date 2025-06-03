import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, title='Matrice de confusion'):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vérité terrain')
    ax.set_title(title)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes, rotation=0)
    st.pyplot(fig)

def app():
    st.header("📂 Entraînement des modèles")

    if 'cleaned_df' not in st.session_state:
        st.warning("Veuillez d'abord effectuer le nettoyage dans l'onglet Traitement.")
        return

    df = st.session_state['cleaned_df']

    if "target" not in df.columns:
        st.error("La colonne 'target' doit être présente dans les données.")
        return

    X = df.drop(columns=["target"])
    y = df["target"]

    # Encodage label
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

    st.write(f"Données train : {len(X_train)}, test : {len(X_test)}")

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Régression Logistique": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True)
    }
    selected = st.multiselect("Choisir modèles à entraîner", list(models.keys()), default=list(models.keys()))

    if st.button("Lancer l'entraînement"):
        results = {}
        for name in selected:
            model = models[name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            results[name] = {
                "model": model,
                "report": report,
                "confusion_matrix": cm,
                "y_pred": y_pred,
            }
            st.success(f"Modèle {name} entraîné.")

        # Sauvegarde résultats
        st.session_state['models'] = {k: v['model'] for k, v in results.items()}
        st.session_state['reports'] = {k: v['report'] for k, v in results.items()}
        st.session_state['cms'] = {k: v['confusion_matrix'] for k, v in results.items()}
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['algo_names'] = selected
        st.session_state['class_names'] = class_names

        st.success("✅ Entraînement terminé. Vous pouvez maintenant aller à l'onglet Évaluation.")

    # Affichage rapports si déjà entraîné
    if 'reports' in st.session_state:
        for name in st.session_state['algo_names']:
            st.write(f"### Rapport pour {name}")
            report_df = pd.DataFrame(st.session_state['reports'][name]).transpose()
            st.dataframe(report_df)

            cm = st.session_state['cms'][name]
            plot_confusion_matrix(cm, classes=st.session_state['class_names'], title=f"Matrice de confusion - {name}")
