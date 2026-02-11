import streamlit as st
import joblib
import numpy as np
from PIL import Image
import streamlit.components.v1 as components

# --- CONFIGURATION ---
st.set_page_config(page_title="BankGuard AI", page_icon="🛡️")

@st.cache_resource
def load_assets():
    model = joblib.load("modele.pkl")
    # Détecter combien de features le modèle attend
    # Souvent stocké dans n_features_in_ pour sklearn
    try:
        expected_features = model.n_features_in_
    except:
        expected_features = 16384 # Valeur par défaut si indétectable
    return model, expected_features

model, n_expected = load_assets()

st.title("🛡️ BankGuard AI")
input_image = st.camera_input("Scanner le billet")

if input_image:
    # 1. Charger l'image
    img = Image.open(input_image).convert('L') # Mode Gris
    
    # 2. CALCULER LA BONNE TAILLE AUTOMATIQUEMENT
    # Si n_expected = 4096, alors la taille est racine(4096) = 64
    size = int(np.sqrt(n_expected)) 
    
    # Redimensionnement précis
    img_resized = img.resize((size, size))
    features = np.array(img_resized).flatten().reshape(1, -1) / 255.0

    # Vérification de sécurité avant de prédire
    if features.shape[1] != n_expected:
        st.error(f"Erreur de dimension : Le modèle attend {n_expected} pixels, mais a reçu {features.shape[1]}.")
    else:
        try:
            prediction = model.predict(features)
            is_real = prediction[0] == 1 # À ajuster selon votre modèle (0 ou 1)

            # --- AFFICHAGE DES RÉSULTATS (COULEURS GÉANTES) ---
            if is_real:
                st.markdown(f'<div style="background-color:#2ecc71; padding:50px; border-radius:20px; text-align:center; color:white;"><h1>✅ BON BILLET</h1></div>', unsafe_allow_html=True)
                components.html('<script>var m = new SpeechSynthesisUtterance("C\'est un bon billet"); m.lang="fr-FR"; window.speechSynthesis.speak(m);</script>', height=0)
            else:
                st.markdown(f'<div style="background-color:#e74c3c; padding:50px; border-radius:20px; text-align:center; color:white;"><h1>❌ FAUX BILLET</h1></div>', unsafe_allow_html=True)
                components.html('<script>var m = new SpeechSynthesisUtterance("Attention, faux billet"); m.lang="fr-FR"; window.speechSynthesis.speak(m);</script>', height=0)
        
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
