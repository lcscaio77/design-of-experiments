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
    .stApp {
        background: linear-gradient(to right,rgb(252, 252, 252), #CFE2F3); /* Bleu pastel clair → Bleu encore plus clair */
        height: 100vh;
    }
    </style>
    """,
    unsafe_allow_html=True
) #jolie fond :)

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




