# Utiliser une image Python de base
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requirements.txt et app.py dans le conteneur
COPY requirements.txt requirements.txt
COPY app.py app.py

# Installer les dépendances
RUN pip install -r requirements.txt

# Exposer le port 8501 pour Streamlit
EXPOSE 8501

# Démarrer Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
