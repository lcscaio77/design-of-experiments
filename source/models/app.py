#--------------------------------Librairies-------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from pyDOE2 import fullfact
from fonctions import *
#-------------------------------Application-------------------------------------
st.title("Design of experiments 🚀")
st.write("### Bienvenue dans l'application de Design of Experiments (DOE) !")
st.write("Cette application vous permet de générer des essais aléatoires, de charger des résultats expérimentaux et d'optimiser les paramètres de votre expérience.")
st.write("Pour commencer, veuillez suivre les étapes ci-dessous :")
#------------------ Etape 1 --------------------

num_params, num_levels, num_trials, target_variable = config_initiale()

#------------------ Etape 2 --------------------
st.write("### Toute les combinaisons possibles :")
df = load_data(num_params, num_levels, num_trials, target_variable)

#------------------ Etape 3 --------------------
tirage_aleatoire(df, num_trials)