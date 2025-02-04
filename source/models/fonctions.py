
#------------------ Importation des librairies ------------------#
import streamlit as st
import pandas as pd
import numpy as np
from pyDOE2 import fullfact
import seaborn as sns
import matplotlib as plt 
#---------------------------- Fonctions -------------------------#

#------------------ Etape 1 --------------------#

def config_initiale():
    st.header("1. Configuration Initiale 🌘")
    num_params = st.number_input("Nombre de paramètres", min_value=1, step=1, value=3)
    num_levels = st.text_input("Niveaux des paramètres (séparés par des virgules, défaut=2)", "2")
    num_trials = st.number_input("Nombre d’essais réalisables", min_value=1, step=1, value=5)
    target_variable = st.text_input("Nom de la variable cible", "Résultat")
    return num_params, num_levels, num_trials, target_variable

#------------------ Etape 2 --------------------#

def load_data(num_params, num_levels, num_trials, target_variable):
    st.header("2. Remplissage des données 🌗")
    upload_option = st.radio("Choisissez une méthode :", ("Remplissage manuel", "Importer un fichier CSV/Excel"))

    if upload_option == "Importer un fichier CSV/Excel":
        uploaded_file = st.file_uploader("Chargez un fichier CSV ou Excel", type=["csv", "xlsx"])  
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        else:
            st.error("Veuillez charger un fichier valide 😑.")
            return None
    else:
        levels = int(num_levels.strip().split(',')[0])  # Supposons un seul niveau pour tous les facteurs
        design_matrix = fullfact([levels] * num_params)

        # Normalisation entre -1 et 1
        design_matrix = -1 + 2 * (design_matrix / (levels - 1))

        df = pd.DataFrame(design_matrix, columns=[f'Facteur {i+1}' for i in range(num_params)])
        df[target_variable] = None  # Ajout de la colonne cible

    return st.data_editor(df, num_rows="dynamic")



#------------------ Etape 3 : Tirage aléatoire --------------------#

import pandas as pd
import itertools

def generate_interactions(df):

    numeric_cols = df.select_dtypes(include=['number']).columns  
    
    for r in range(2, len(numeric_cols) + 1):  
        for cols in itertools.combinations(numeric_cols, r):
            col_name = "_x_".join(cols)  # Ex: "X1_x_X2"
            df[col_name] = df[list(cols)].prod(axis=1)  
            
    return df #cette fonction permet de visualiser les confusions avec les interactions entre variables

def random_sample(df, num_trials):

    st.header("3. Tirage Aléatoire et Confusions 🌖")

    if st.button("Générer des essais aléatoires"):
        if num_trials > df.shape[0]:
            st.error("Le nombre d'essais demandés dépasse le nombre de lignes disponibles.")
            return None
        
        randomized_trials = df.sample(n=num_trials, replace=False).reset_index(drop=True)  # Tirage aléatoire

        st.write("### Essais aléatoires générés :")
        
        edited_trials = st.data_editor(randomized_trials, num_rows="dynamic")  

        csv = edited_trials.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les essais sélectionnés",
            data=csv,
            file_name="essais_aleatoires.csv",
            mime="text/csv"
        )

        return generate_interactions(edited_trials.drop(columns=['Résultat']))
    return None  # Si aucun bouton n'est cliqué, retourne None


#------------------ Etape 3.bis : Matrice de confusion --------------------#

def df_confusion(df):
    if df is None or df.empty:
        st.warning("Aucune donnée disponible pour générer la matrice de confusion.")
        return None  # Sécurité

    n = df.shape[1]
    confusion_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            confusion_matrix[i, j] = np.sum(df.iloc[:, i] == df.iloc[:, j])  

    confusion_matrix /= df.shape[0]

    labels = df.columns if df.columns is not None else [f"Col {i+1}" for i in range(n)]
    confusion_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

    st.write("### Matrice de Confusion Normalisée")
    st.dataframe(confusion_df.style.format(precision=2).background_gradient(cmap="Blues"))

    return confusion_df


















