import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger les modèles
regression_model = joblib.load("model.joblib")
classification_model = joblib.load("classification_model.joblib")

# Charger les 5 variables les plus importantes
feature_names = ["grade", "sqft_living", "lat", "long", "sqft_living15"]  

# Interface utilisateur
st.title("Prédiction de la Valeur des Maisons")

option = st.radio("Choisissez un mode :", ("Régression", "Classification"))

# Saisie utilisateur
data_input = {}
for feature in feature_names:
    data_input[feature] = st.number_input(f"{feature}", value=0.0)

# Transformation des entrées en dataframe
input_data = pd.DataFrame([data_input])

if st.button("Prédire"):
    if option == "Régression":
        prediction = regression_model.predict(input_data)
        st.success(f"Prix estimé : {prediction[0]:,.2f} USD")
    else:
        classification = classification_model.predict(input_data)
        result = "Élevé" if classification[0] == 1 else "Bas"
        st.success(f"Catégorie de prix : {result}")
