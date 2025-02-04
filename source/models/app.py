#--------------------------------Librairies-------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from pyDOE2 import fullfact
from fonctions import *

#-------------------------------Application-------------------------------------
st.title("Design of Experiments 🚀")
st.write("### Bienvenue dans l'application de Design of Experiments (DOE) !")
st.write("Cette application vous permet de générer des essais aléatoires, de charger des résultats expérimentaux et d'optimiser les paramètres de votre expérience.")
st.write("Pour commencer, veuillez suivre les étapes ci-dessous :")

#------------------ Étape 1 --------------------
num_params, num_levels, num_trials, target_variable = config_initiale()

#------------------ Étape 2 --------------------
st.write("### Toutes les combinaisons possibles :")
df = load_data(num_params, num_levels, num_trials, target_variable)

#------------------ Étape 3 --------------------
selected_trials = random_sample(df, num_trials)

if selected_trials is not None:  # Debugging pour la generation confusion 
    df_confusion(selected_trials)
