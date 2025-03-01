#!/bin/bash

# venv
echo "Création de l'environnement virtuel..."
python3 -m venv venv
source venv/bin/activate
echo "Installation des dépendances..."
pip install -r requirements.txt


echo "Lancement de l'application..."
streamlit run app.py



#Pour linux macos
#chmod +x lancez_moi.sh
# ./lancez_moi.sh


