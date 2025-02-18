#------------------ Importation des librairies ------------------#
import streamlit as st
import pandas as pd
import numpy as np
from pyDOE2 import fullfact
import seaborn as sns
import matplotlib as plt 
import itertools
import statsmodels.api as sm
#---------------------------- Fonctions -------------------------#

#------------------ Etape 1 --------------------#

def config_initiale():
    st.header("1. Configuration Initiale 🌘")
    num_params = st.number_input("Nombre de paramètres", min_value=1, step=1, value=3)
    
    # Configuration des niveaux
    niveau_config = st.radio(
        "Comment souhaitez-vous configurer les niveaux ?",
        ("Niveaux identiques pour tous les facteurs", "Niveaux différents par facteur")
    )

    if niveau_config == "Niveaux identiques pour tous les facteurs":
        num_levels = st.text_input("Niveaux des paramètres (par défaut = 2)", "2")
        try:
            levels = int(num_levels.strip().split(',')[0])
            levels_list = [levels] * num_params
        except ValueError:
            st.error("Veuillez entrer un nombre valide pour les niveaux")
            return None, None, None, None
    else:
        st.write("### Configuration des niveaux par paramètre")
        levels_list = []
        cols = st.columns(min(num_params, 4))
        for i in range(num_params):
            with cols[i % 4]:
                level = st.number_input(
                    f"Niveaux Facteur {i+1}", 
                    min_value=2, 
                    value=2, 
                    key=f"level_{i}"
                )
                levels_list.append(level)
        num_levels = ','.join(map(str, levels_list))  # Conversion en format texte pour compatibilité

    num_trials = st.number_input("Nombre d'essais réalisables", min_value=1, step=1, value=5)
    target_variable = st.text_input("Nom de la variable cible", "Résultat")
    
    return num_params, num_levels, num_trials, target_variable

#------------------ Etape 2 --------------------#

def load_data(num_params, num_levels, num_trials, target_variable):
    st.header("2. Remplissage des données 🌗")
    
    upload_option = st.radio("Choisissez une méthode :", ("Remplissage manuel", "Importer un fichier CSV/Excel"))

    if upload_option == "Importer un fichier CSV/Excel":
        uploaded_file = st.file_uploader("Chargez un fichier CSV ou Excel", type=["csv", "xlsx"])  
        if uploaded_file is not None:
            try:
                # Lecture directe en DataFrame pandas
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                
                # Vérification de la structure du fichier
                expected_columns = [f'Facteur {i+1}' for i in range(num_params)] + [target_variable]
                missing_columns = [col for col in expected_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Colonnes manquantes dans le fichier : {', '.join(missing_columns)}")
                    st.info(f"Le fichier doit contenir les colonnes : {', '.join(expected_columns)}")
                    return None
                
                st.success("Données importées avec succès!")
                st.write("### Aperçu des données expérimentales :")
                st.dataframe(df)  # Affichage simple du DataFrame
                return df  # Retourne le DataFrame pandas directement
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier : {str(e)}")
                return None
    else:
        try:
            # Conversion de num_levels en liste
            levels_list = [int(level.strip()) for level in num_levels.split(',')]
            if len(levels_list) == 1:
                levels_list = levels_list * num_params
            
            # Création de la matrice de design avec les niveaux spécifiés
            design_matrix = fullfact(levels_list)
            
            # Normalisation individuelle pour chaque facteur
            normalized_matrix = np.zeros_like(design_matrix, dtype=float)
            for col in range(design_matrix.shape[1]):
                normalized_matrix[:, col] = -1 + 2 * (design_matrix[:, col] / (levels_list[col] - 1))
            
            # Création du DataFrame
            df = pd.DataFrame(
                normalized_matrix, 
                columns=[f'Facteur {i+1}' for i in range(num_params)]
            )
            df[target_variable] = None

            # Affichage des niveaux choisis
            st.write("### Niveaux choisis pour chaque facteur :")
            niveau_info = {f"Facteur {i+1}": level for i, level in enumerate(levels_list)}
            st.write(niveau_info)
            
            edited_df = st.data_editor(df, num_rows="dynamic")
            return edited_df  # Retourne le DataFrame édité
            
        except Exception as e:
            st.error(f"Erreur lors de la création de la matrice : {str(e)}")
            return None

#------------------ Etape 3 : Tirage aléatoire --------------------#



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

#------------------ Etape 4 : Régression Linéaire --------------------#
# À implémenter : Analyse par régression linéaire des résultats expérimentaux

def otpimisation(df):
    features = df.columns[:-1]
    x = df[features]
    y = df['Résultat']
    x = sm.add_constant(x)

    resultat = sm.OLS(y, x).fit()

    return resultat.summary()



def regression_lineaire(df):
    st.header("4. Analyse par Régression Linéaire 🌕")
    
    if st.button("Effectuer l'analyse"):
        if df is None or df.empty:
            st.warning("Aucune donnée disponible pour l'analyse.")
        else:
            st.write("### Analyse par Régression Linéaire")
            st.write("### Résultats de la régression linéaire :")
            st.write(otpimisation(df))















