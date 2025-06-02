import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    st.header("🔍 Exploration des données")
    df = pd.read_csv("data/vin.csv")
    st.write(df.head())

    st.subheader("Distribution d'une variable")
    colonne = st.selectbox("Choisissez une variable numérique :", df.select_dtypes('number').columns)
    fig, ax = plt.subplots()
    sns.histplot(df[colonne], kde=True, ax=ax)
    st.pyplot(fig)
