#--------------------------------Librairies-------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from pyDOE2 import fullfact
from fonctions import *

#-------------------------------Application-------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #e1f9fc 0%, #ffffff 100%);
        font-family: 'Montserrat', sans-serif;
        color: #1e1e1e;
        min-height: 100vh;
        padding: 20px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #57b8c9;
        font-weight: 700;
    }

    .css-18e3th9, .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.85) !important;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        color: #1e1e1e;
    }

    .css-q8sbsg p {
        color: #333333;
    }

    input, textarea, select {
        background-color: #f3fdff !important;
        color: #57b8c9 !important;
        border: 1px solid #57b8c9 !important;
        border-radius: 8px !important;
        padding: 10px;
    }

    ::placeholder {
        color: #888 !important;
    }

    .stButton>button {
        background-color: #57b8c9;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        transition: 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #70d0e0;
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Design of Experiments 🚀")
st.write("### Bienvenue dans l'application de Design of Experiments !")
st.write("Cette application vous permet de générer des essais, de charger des résultats expérimentaux et d'optimiser les paramètres de votre expérience.")
st.write("Veuillez suivre les étapes ci-dessous :")

#------------------ Étape 1 --------------------
num_params, num_levels, num_trials, target_variable = config_initiale()

#------------------ Étape 2 --------------------
st.write("### Toutes les combinaisons possibles :")
df = load_data(num_params, num_levels, num_trials, target_variable)

#------------------ Étape 3 --------------------
selected_trials = random_sample_lhs(df, num_trials)

if selected_trials is not None:  # Debugging pour la generation confusion 
    df_confusion(selected_trials)  

#------------------ Étape 4 --------------------
if 'Résultat' in df.columns and not df['Résultat'].isnull().any():
    regression_lineaire(df)
else:
    st.write("### 4. Aucun résultat expérimental chargé")




