import streamlit as st
import numpy as np
import joblib

# Charger le modèle
model = joblib.load("model_final.joblib")  # Assure-toi que ce fichier existe et est bien sans normalisation

# Interface utilisateur
st.title("🏡 Prédiction du Prix des Maisons")
st.write("Entrez les caractéristiques de la maison pour estimer son prix.")

# Variables sélectionnées (doivent correspondre aux variables utilisées pour l'entraînement)
features = ['sqft_living', 'grade', 'sqft_above', 'bathrooms', 'view', 'sqft_living15','sqft_basement']

# Création des champs d'entrée utilisateur
data = {}
for feature in features:
    data[feature] = st.number_input(f"{feature}", value=0.0)

# Bouton de prédiction
if st.button("Prédire le prix"):
    # Transformation des données d'entrée (sans normalisation)
    input_data = np.array([[data[feature] for feature in features]])
    
    # Prédiction
    prediction = model.predict(input_data)

    # Affichage du résultat
    st.success(f"💰 Le prix estimé de la maison est : {prediction[0]:,.2f} $")
