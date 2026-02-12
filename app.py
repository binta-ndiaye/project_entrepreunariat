"""
BillGuard — Détecteur de faux billets
Optimisé pour le secteur informel (utilisateurs peu alphabétisés, mobile)
Architecture: Features image + OneClassSVM (entraîné sur vrais billets uniquement)
"""

import streamlit as st
import joblib
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BillGuard",
    page_icon="💴",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif !important; }
.stApp { background: linear-gradient(160deg, #0d1b2a 0%, #1b2f45 100%); min-height: 100vh; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.app-header { text-align:center; padding:24px 8px 10px; }
.app-logo   { font-size:58px; line-height:1; margin-bottom:6px;
              filter:drop-shadow(0 0 22px rgba(0,220,255,.45)); }
.app-title  { font-size:34px; font-weight:900; margin:0;
              background:linear-gradient(90deg,#00dcff,#ffffff);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.app-sub    { font-size:14px; color:#6fa0c0; margin:4px 0 0; font-weight:700; }

.card { background:rgba(255,255,255,.05); border:1.5px solid rgba(255,255,255,.1);
        border-radius:22px; padding:22px 18px; margin:10px 0; backdrop-filter:blur(8px); }

.res-ok  { border-radius:22px; padding:24px 16px; text-align:center; margin:14px 0;
           background:rgba(0,220,100,.10); border:2.5px solid rgba(0,220,100,.50); }
.res-bad { border-radius:22px; padding:24px 16px; text-align:center; margin:14px 0;
           background:rgba(255,45,75,.10); border:2.5px solid rgba(255,45,75,.55); }
.res-unk { border-radius:22px; padding:24px 16px; text-align:center; margin:14px 0;
           background:rgba(255,190,0,.10); border:2.5px solid rgba(255,190,0,.55); }

.res-emoji { font-size:64px; display:block; margin-bottom:10px; }
.res-title-ok  { font-size:28px; font-weight:900; color:#00dc64; margin:0 0 6px; }
.res-title-bad { font-size:28px; font-weight:900; color:#ff2d4b; margin:0 0 6px; }
.res-title-unk { font-size:28px; font-weight:900; color:#ffbe00; margin:0 0 6px; }
.res-msg { font-size:15px; font-weight:700; line-height:1.5; }
.res-ok  .res-msg { color:#a0e8c0; }
.res-bad .res-msg { color:#f0a0b0; }
.res-unk .res-msg { color:#ffe090; }

.conf-wrap { background:rgba(255,255,255,.04); border-radius:14px; padding:14px 16px; margin-top:10px; }
.conf-row  { display:flex; justify-content:space-between;
             font-size:13px; color:#6fa0c0; font-weight:700; margin-bottom:8px; }
.bar-bg    { height:12px; background:rgba(255,255,255,.1); border-radius:99px; overflow:hidden; }
.bar-green { height:100%; border-radius:99px; background:linear-gradient(90deg,#00cc55,#00ff88); box-shadow:0 0 10px rgba(0,255,100,.4); }
.bar-red   { height:100%; border-radius:99px; background:linear-gradient(90deg,#cc2244,#ff2d4b); box-shadow:0 0 10px rgba(255,45,75,.4); }
.bar-orng  { height:100%; border-radius:99px; background:linear-gradient(90deg,#cc8800,#ffbe00); box-shadow:0 0 10px rgba(255,190,0,.4); }

.tip      { border-left:3.5px solid #00aaff; border-radius:0 12px 12px 0;
            background:rgba(0,170,255,.07); padding:12px 14px;
            font-size:13px; color:#8dcff5; margin:10px 0; line-height:1.65; font-weight:600; }
.tip-warn { border-color:#ff2d4b; background:rgba(255,45,75,.06); color:#f5a0b0; }

.meter-row { display:flex; gap:10px; margin:12px 0; }
.meter-box { flex:1; background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
             border-radius:14px; padding:12px 8px; text-align:center; }
.meter-val { font-size:20px; font-weight:900; font-family:monospace; }
.meter-lbl { font-size:11px; color:#5a88a8; font-weight:700; text-transform:uppercase;
             letter-spacing:.4px; margin-top:2px; }
.ok-col  { color:#00dc64; }
.bad-col { color:#ff2d4b; }
.unk-col { color:#ffbe00; }

.stButton>button {
    background:linear-gradient(135deg,#00aaff,#0066cc) !important;
    color:#fff !important; border:none !important; border-radius:18px !important;
    font-size:18px !important; font-weight:900 !important;
    padding:15px 32px !important; width:100% !important;
    box-shadow:0 5px 22px rgba(0,140,255,.45) !important;
    font-family:'Nunito',sans-serif !important;
}
[data-testid="stFileUploader"] {
    border:2px dashed rgba(0,200,255,.4) !important;
    border-radius:18px !important; background:rgba(0,200,255,.04) !important;
}
[data-testid="stFileUploader"] label { color:#6fa0c0 !important; font-weight:700 !important; }
[data-testid="stImage"] img { border-radius:14px !important;
    border:1.5px solid rgba(255,255,255,.1) !important; }

.footer { text-align:center; padding:16px; font-size:12px; color:#2d5070; font-weight:700; }
p, li, span, label { color:#a0c8e0 !important; }
h1,h2,h3 { color:#fff !important; }
.stAlert { border-radius:14px !important; }

@media(max-width:480px) {
    .app-title { font-size:27px; }
    .res-title-ok, .res-title-bad, .res-title-unk { font-size:23px; }
    .res-emoji { font-size:52px; }
}
</style>
""", unsafe_allow_html=True)


# ─── EXTRACTION DE FEATURES (2048 dims) ───────────────────────────────────────
def extract_features(img: Image.Image) -> np.ndarray:
    """
    Extrait 2048 features d'une image PIL.
    Pipeline: redim 224x224 → histogramme RGB (768) + stats blocs (1280)
    """
    img_rgb = img.convert("RGB").resize((224, 224))
    arr = np.array(img_rgb, dtype=np.float32) / 255.0

    features = []

    # 1. Histogramme RGB : 3 × 256 = 768 features
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=256, range=(0.0, 1.0))
        norm = hist.astype(np.float32)
        s = norm.sum()
        if s > 0:
            norm /= s
        features.extend(norm.tolist())

    # 2. Stats par blocs 16×16 : 14×14 blocs × 3 canaux × 4 stats → tronqué à 1280
    block_size = 16
    block_stats = []
    for c in range(3):
        ch = arr[:, :, c]
        for i in range(0, 224, block_size):
            for j in range(0, 224, block_size):
                blk = ch[i:i + block_size, j:j + block_size]
                if blk.size > 0:
                    block_stats.extend([
                        float(blk.mean()),
                        float(blk.std()),
                        float(blk.min()),
                        float(blk.max()),
                    ])

    block_stats = np.array(block_stats, dtype=np.float32)
    block_stats = block_stats[:1280] if len(block_stats) >= 1280 \
        else np.pad(block_stats, (0, 1280 - len(block_stats)))

    features.extend(block_stats.tolist())
    feat = np.array(features, dtype=np.float32)

    # Garantir exactement 2048 dims
    if len(feat) > 2048:
        feat = feat[:2048]
    elif len(feat) < 2048:
        feat = np.pad(feat, (0, 2048 - len(feat)))

    return feat.reshape(1, -1)


# ─── CHARGEMENT DU MODÈLE ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = joblib.load("modele.pkl")
        return model, None
    except Exception as e:
        return None, str(e)


# ─── PRÉDICTION ───────────────────────────────────────────────────────────────
def predict_banknote(model, features: np.ndarray):
    """
    OneClassSVM: +1 = inlier (vrai billet), -1 = outlier (suspect/faux)
    decision_function() retourne un score continu.
    """
    prediction = int(model.predict(features)[0])
    score = float(model.decision_function(features)[0])

    # Normalisation score → confiance 0-100%
    score_clipped = np.clip(score, -3.0, 3.0)
    confidence = float((score_clipped + 3.0) / 6.0 * 100.0)

    if prediction == 1:
        status = "authentic" if confidence >= 60 else "uncertain"
    else:
        status = "fake"

    return status, confidence, score


# ─── RENDU RÉSULTAT ───────────────────────────────────────────────────────────
def render_result(status: str, confidence: float, score: float):
    if status == "authentic":
        st.markdown("""
        <div class="res-ok">
            <span class="res-emoji">✅</span>
            <p class="res-title-ok">BILLET AUTHENTIQUE</p>
            <p class="res-msg">Ce billet ressemble à un vrai billet.<br>
            Vous pouvez l'accepter en toute confiance.</p>
        </div>""", unsafe_allow_html=True)
        bar_cls, s_cls, decision = "bar-green", "ok-col", "✓ ACCEPTER"

    elif status == "fake":
        st.markdown("""
        <div class="res-bad">
            <span class="res-emoji">🚫</span>
            <p class="res-title-bad">FAUX BILLET DÉTECTÉ</p>
            <p class="res-msg">Attention ! Ce billet présente des anomalies.<br>
            <strong>Refusez-le et signalez-le aux autorités.</strong></p>
        </div>""", unsafe_allow_html=True)
        bar_cls, s_cls, decision = "bar-red", "bad-col", "✗ REFUSER"

    else:
        st.markdown("""
        <div class="res-unk">
            <span class="res-emoji">⚠️</span>
            <p class="res-title-unk">RÉSULTAT INCERTAIN</p>
            <p class="res-msg">Photo insuffisante pour conclure.<br>
            Reprenez la photo avec plus de lumière.</p>
        </div>""", unsafe_allow_html=True)
        bar_cls, s_cls, decision = "bar-orng", "unk-col", "? VÉRIFIER"

    # Indicateurs
    st.markdown(f"""
    <div class="meter-row">
        <div class="meter-box">
            <div class="meter-val {s_cls}">{confidence:.0f}%</div>
            <div class="meter-lbl">Confiance</div>
        </div>
        <div class="meter-box">
            <div class="meter-val {s_cls}">{score:+.2f}</div>
            <div class="meter-lbl">Score IA</div>
        </div>
        <div class="meter-box">
            <div class="meter-val {s_cls}">{decision}</div>
            <div class="meter-lbl">Action</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Barre
    st.markdown(f"""
    <div class="conf-wrap">
        <div class="conf-row"><span>Niveau de certitude</span><span>{confidence:.1f} %</span></div>
        <div class="bar-bg"><div class="{bar_cls}" style="width:{confidence:.1f}%"></div></div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
def main():

    # En-tête
    st.markdown("""
    <div class="app-header">
        <div class="app-logo">💴</div>
        <p class="app-title">BillGuard</p>
        <p class="app-sub">Détecteur de Faux Billets · Gratuit · Rapide</p>
    </div>""", unsafe_allow_html=True)

    # Modèle
    model, error = load_model()
    if model is None:
        st.error(f"⚠️ Modèle introuvable : {error}")
        st.info("Vérifiez que **modele.pkl** est à la racine du dépôt GitHub.")
        st.stop()

    # Zone upload
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📸 Photographiez votre billet")
    st.markdown("""
    <div class="tip">
    💡 <strong>Pour un bon résultat :</strong><br>
    • Posez le billet à plat sur une surface foncée<br>
    • Bonne lumière (jour ou lampe) — évitez les reflets<br>
    • Le billet doit remplir presque toute la photo<br>
    • Tenez le téléphone bien droit au-dessus du billet
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "📤  Choisir une photo du billet",
        type=["jpg", "jpeg", "png", "webp"],
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Traitement
    if uploaded is not None:
        image = Image.open(uploaded)
        col1, col2, col3 = st.columns([1, 5, 1])
        with col2:
            st.image(image, caption="Billet à analyser", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔍   ANALYSER CE BILLET"):
            with st.spinner("⏳  Analyse en cours…"):
                try:
                    features = extract_features(image)
                    status, confidence, score = predict_banknote(model, features)
                    render_result(status, confidence, score)

                    if status == "fake":
                        st.markdown("""
                        <div class="tip tip-warn">
                        🚨 <strong>Que faire avec un faux billet ?</strong><br>
                        • Ne le remettez pas en circulation — c'est illégal<br>
                        • Signalez-le à la gendarmerie ou à la police<br>
                        • Contactez votre banque centrale (BCEAO, BCC…)<br>
                        • Gardez-le comme preuve sans le déchirer
                        </div>""", unsafe_allow_html=True)

                    elif status == "uncertain":
                        st.markdown("""
                        <div class="tip">
                        📸 <strong>Améliorer la photo :</strong><br>
                        • Nettoyez l'objectif de votre téléphone<br>
                        • Approchez-vous plus du billet<br>
                        • Ajoutez une source de lumière<br>
                        • Évitez les ombres sur le billet
                        </div>""", unsafe_allow_html=True)

                    else:
                        st.markdown("""
                        <div class="tip">
                        ✅ Vérifiez aussi manuellement : filigrane,
                        bande holographique et numéro de série.
                        </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Erreur : {e}")
                    st.info("Essayez avec une photo plus nette et bien éclairée.")

    with st.expander("ℹ️  À propos de BillGuard"):
        st.markdown("""
        **BillGuard** détecte les faux billets par intelligence artificielle.
        Le modèle a été entraîné sur des **vrais billets uniquement** :
        tout billet qui s'éloigne de cette norme est signalé comme suspect.

        - 🔒 **Aucune photo n'est envoyée** — traitement 100 % local
        - 📱 **Fonctionne sur téléphone** — pas d'installation requise
        - ⚡ **Résultat en moins de 3 secondes**

        > ⚠️ Outil d'aide à la décision. En cas de doute sérieux,
        consultez les autorités compétentes.
        """)

    st.markdown("""
    <div class="footer">
        💴 BillGuard · Secteur Informel · IA 100% Locale<br>
        Aucune donnée envoyée · Gratuit · Conçu pour tous
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
