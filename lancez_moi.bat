@echo off
echo Creation et activation de l'environnement virtuel...

:: Créer l'environnement virtuel s'il n'existe pas
if not exist "venv" (
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

:: Lancer l'application
echo Lancement de l'application...
python -m streamlit run source/models/app.py

pause 