import streamlit as st
import joblib
import numpy as np
from PIL import Image
import streamlit.components.v1 as components

# --- CONFIGURATION INITIALE ---
st.set_page_config(page_title="BankGuard AI", page_icon="🛡️", layout="centered")

# --- DESIGN INNOVANT ET ACCESSIBLE (CSS) ---
st.markdown("""
    <style>
    /* Masquer les menus Streamlit pour faire "App Mobile" */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Fond sombre professionnel */
    .stApp { background-color: #0E1117; }

    /* Conteneur de résultat géant */
    .result-card {
        padding: 50px 20px;
        border-radius: 30px;
        text-align: center;
        margin-top: 20px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    .status-icon { font-size: 120px; margin-bottom: 10px; }
    .status-text { font-size: 35px; font-weight: 900; text-transform: uppercase; }
    
    /* Bouton caméra personnalisé */
    [data-testid="stCameraInput"] {
        border: 4px dashed #38bdf8;
        border-radius: 25px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE DE CHARGEMENT ---
@st.cache_resource
def load_assets():
    model = joblib.load("modele.pkl")
    return model

try:
    model = load_assets()
except:
    st.error("⚠️ Erreur : Placez le fichier 'modele.pkl' dans le dossier.")
    st.stop()

# --- EN-TÊTE VISUELLE ---
st.markdown("<h1 style='text-align: center; color: #38bdf8;'>🛡️ BankGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Placez le billet devant la caméra et validez</p>", unsafe_allow_html=True)

# --- CAPTURE IMAGE ---
input_image = st.camera_input("") # Label vide pour épurer l'UI

if input_image:
    # Traitement de l'image (Simulation de la taille attendue par votre modèle)
    img = Image.open(input_image).convert('L') # Gris
    img = img.resize((128, 128)) # Ajustez à la taille de votre entraînement
    features = np.array(img).flatten().reshape(1, -1) / 255.0
    
    # Prédiction
    prediction = model.predict(features)
    # On suppose : 1 = Vrai, 0 = Faux
    is_real = prediction[0] == 1 

    # --- AFFICHAGE INNOVANT DU RÉSULTAT ---
    if is_real:
        # DESIGN VERT (AUTHENTIQUE)
        st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, #1e7e34, #28a745);">
                <div class="status-icon">✅</div>
                <div class="status-text">BON BILLET</div>
                <div style="font-size: 20px;">Le billet est vrai</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Audio et Vibration (Vrai)
        components.html("""
            <script>
            var msg = new SpeechSynthesisUtterance("C'est un bon billet. Vous pouvez l'accepter.");
            msg.lang = 'fr-FR';
            window.speechSynthesis.speak(msg);
            if (window.navigator.vibrate) window.navigator.vibrate(200);
            </script>
        """, height=0)
        st.balloons()

    else:
        # DESIGN ROUGE (ALERTE)
        st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, #bd2130, #dc3545);">
                <div class="status-icon">🚨</div>
                <div class="status-text">ATTENTION !</div>
                <div style="font-size: 20px;">BILLET DOUTEUX</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Audio et Vibration (Faux)
        components.html("""
            <script>
            var msg = new SpeechSynthesisUtterance("Attention ! Ce billet semble faux. Soyez prudent.");
            msg.lang = 'fr-FR';
            window.speechSynthesis.speak(msg);
            // Vibration d'alerte (3 fois)
            if (window.navigator.vibrate) window.navigator.vibrate([500, 200, 500, 200, 500]);
            </script>
        """, height=0)

# --- GUIDE SANS TEXTE (PICTO) ---
st.markdown("<br><hr>", unsafe_allow_html=True)
cols = st.columns(3)
with cols[0]: st.markdown("☀️<br><small>Bonne lumière</small>", unsafe_allow_html=True)
with cols[1]: st.markdown("📏<br><small>Bien cadré</small>", unsafe_allow_html=True)
with cols[2]: st.markdown("📱<br><small>Sans bouger</small>", unsafe_allow_html=True)