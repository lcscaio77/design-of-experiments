#!/bin/bash

echo "Creation et activation de l'environnement virtuel..."

# Créer l'environnement virtuel s'il n'existe pas
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Lancer l'application
echo "Lancement de l'application..."
python -m streamlit run source/models/app.py



#Pour linux macos
#chmod +x lancez_moi.sh
# ./lancez_moi.sh


