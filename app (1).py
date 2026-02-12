"""
BillGuard — Détecteur de faux billets FCFA
Optimisé secteur informel · Recadrage automatique du billet · IA 100% locale
"""

import streamlit as st
import joblib
import numpy as np
from PIL import Image
import warnings
import cv2

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
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif !important; }
.stApp { background: #0a1628; min-height: 100vh; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 28px 12px 16px;
}
.app-logo {
    font-size: 72px;
    line-height: 1;
    margin-bottom: 8px;
    display: block;
    filter: drop-shadow(0 0 28px rgba(0, 220, 120, .5));
    animation: pulse-logo 2.5s ease-in-out infinite;
}
@keyframes pulse-logo {
    0%, 100% { transform: scale(1); filter: drop-shadow(0 0 18px rgba(0,220,120,.4)); }
    50%       { transform: scale(1.06); filter: drop-shadow(0 0 36px rgba(0,220,120,.7)); }
}
.app-title {
    font-size: 36px;
    font-weight: 900;
    margin: 0;
    background: linear-gradient(90deg, #00ff88, #00ccff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.app-tagline {
    font-size: 15px;
    color: #4a7a9b;
    margin: 6px 0 0;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* ── Instruction visuelle ── */
.how-to {
    display: flex;
    gap: 10px;
    margin: 18px 0;
    justify-content: center;
}
.how-step {
    flex: 1;
    background: rgba(255,255,255,.04);
    border: 1.5px solid rgba(255,255,255,.08);
    border-radius: 18px;
    padding: 14px 8px;
    text-align: center;
    max-width: 100px;
}
.how-icon { font-size: 30px; display: block; margin-bottom: 5px; }
.how-txt  { font-size: 11px; color: #5a8aaa; font-weight: 700; line-height: 1.4; }

/* ── Upload zone ── */
.upload-card {
    background: rgba(0, 200, 100, .04);
    border: 2px dashed rgba(0, 200, 100, .3);
    border-radius: 22px;
    padding: 20px 16px;
    margin: 14px 0;
    transition: border-color .3s;
}
[data-testid="stFileUploader"] {
    border: none !important;
    background: transparent !important;
}
[data-testid="stFileUploader"] label { color: #4a8a6a !important; font-weight: 700 !important; font-size: 16px !important; }

/* ── Qualité photo ── */
.quality-bar {
    background: rgba(255,255,255,.05);
    border-radius: 14px;
    padding: 12px 14px;
    margin: 10px 0;
}
.q-label { font-size: 12px; color: #4a7a9b; font-weight: 700; text-transform: uppercase;
           letter-spacing: .5px; margin-bottom: 6px; display: flex; justify-content: space-between; }
.bar-bg  { height: 10px; background: rgba(255,255,255,.08); border-radius: 99px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; transition: width .5s ease; }
.bar-green  { background: linear-gradient(90deg, #00cc55, #00ff88); }
.bar-orange { background: linear-gradient(90deg, #cc7700, #ffaa00); }
.bar-red    { background: linear-gradient(90deg, #cc2244, #ff4466); }

/* ── Badge recadrage ── */
.crop-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 200, 100, .12);
    border: 1px solid rgba(0, 200, 100, .35);
    border-radius: 99px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 700;
    color: #00cc66;
    margin: 6px 0;
}
.crop-badge-warn {
    background: rgba(255, 170, 0, .12);
    border-color: rgba(255, 170, 0, .35);
    color: #ffaa00;
}

/* ── Bouton ── */
.stButton > button {
    background: linear-gradient(135deg, #00cc55, #009944) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 20px !important;
    font-size: 20px !important;
    font-weight: 900 !important;
    padding: 18px 32px !important;
    width: 100% !important;
    box-shadow: 0 6px 28px rgba(0, 200, 80, .45) !important;
    font-family: 'Nunito', sans-serif !important;
    letter-spacing: .3px !important;
    transition: transform .15s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

/* ── Résultats ── */
.res-card {
    border-radius: 24px;
    padding: 28px 20px;
    text-align: center;
    margin: 16px 0;
    animation: fadeInUp .4s ease both;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.res-ok  { background: rgba(0,220,100,.09);  border: 2.5px solid rgba(0,220,100,.45); }
.res-bad { background: rgba(255,45,75,.09);  border: 2.5px solid rgba(255,45,75,.50); }
.res-unk { background: rgba(255,190,0,.09);  border: 2.5px solid rgba(255,190,0,.50); }

.res-emoji { font-size: 72px; display: block; margin-bottom: 12px;
             animation: bounce-in .5s cubic-bezier(.34,1.56,.64,1) both; }
@keyframes bounce-in {
    from { transform: scale(.4); opacity: 0; }
    to   { transform: scale(1);  opacity: 1; }
}
.res-title { font-size: 30px; font-weight: 900; margin: 0 0 8px; }
.res-ok   .res-title { color: #00ff77; }
.res-bad  .res-title { color: #ff3355; }
.res-unk  .res-title { color: #ffcc00; }
.res-msg { font-size: 15px; font-weight: 700; line-height: 1.6; }
.res-ok  .res-msg { color: #80e8b0; }
.res-bad .res-msg { color: #f0a0b0; }
.res-unk .res-msg { color: #ffe090; }

/* ── Métriques ── */
.metrics-row { display: flex; gap: 10px; margin: 14px 0; }
.metric-box  { flex: 1; background: rgba(255,255,255,.04);
               border: 1px solid rgba(255,255,255,.07);
               border-radius: 16px; padding: 14px 8px; text-align: center; }
.metric-val { font-size: 22px; font-weight: 900; font-family: monospace; }
.metric-lbl { font-size: 10px; color: #3a6a8a; font-weight: 700;
              text-transform: uppercase; letter-spacing: .4px; margin-top: 3px; }
.c-green { color: #00ff77; }
.c-red   { color: #ff3355; }
.c-orng  { color: #ffcc00; }

/* ── Tip ── */
.tip {
    border-left: 3px solid #00aaff;
    border-radius: 0 14px 14px 0;
    background: rgba(0,170,255,.07);
    padding: 12px 14px;
    font-size: 13px;
    color: #7abfe8;
    margin: 10px 0;
    line-height: 1.7;
    font-weight: 600;
}
.tip-danger { border-color: #ff3355; background: rgba(255,45,75,.07); color: #f0a0b0; }
.tip-warn   { border-color: #ffcc00; background: rgba(255,200,0,.07); color: #ffe090; }

/* ── Footer ── */
.footer { text-align: center; padding: 20px; font-size: 12px; color: #1e3a5a; font-weight: 700; margin-top: 8px; }

/* ── Overrides ── */
p, li, span, label { color: #7aaec8 !important; }
h1, h2, h3 { color: #e0f0ff !important; }
.stAlert { border-radius: 16px !important; }
[data-testid="stImage"] img { border-radius: 16px !important; border: 1.5px solid rgba(255,255,255,.1) !important; }

@media(max-width: 480px) {
    .app-title  { font-size: 30px; }
    .res-title  { font-size: 24px; }
    .res-emoji  { font-size: 58px; }
    .how-step   { padding: 10px 5px; }
    .how-icon   { font-size: 24px; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  RECADRAGE AUTOMATIQUE DU BILLET
# ═══════════════════════════════════════════════════════════════════════════════
def auto_crop_banknote(pil_img: Image.Image, margin: float = 0.02):
    """
    Détecte et recadre automatiquement le billet dans la photo.

    Stratégie :
    1. Convertit en BGR (OpenCV)
    2. Flou gaussien + détection de contours (Canny)
    3. Trouve le plus grand contour rectangulaire
    4. Perspective warp si angle > 5°, sinon crop simple
    5. Retourne (image_recadrée, was_detected: bool)
    """
    img_np = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    # 1. Préparation
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    # 2. Trouver contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return pil_img, False

    # Trier par aire décroissante
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    img_area = h * w

    best_quad = None
    for cnt in cnts[:8]:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.12:   # trop petit (< 12% de l'image)
            continue
        if area > img_area * 0.98:   # trop grand (presque toute l'image)
            continue

        # Approximation polygonale
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            best_quad = approx
            break

    if best_quad is not None:
        pts = best_quad.reshape(4, 2).astype(np.float32)

        # Ordonner : haut-gauche, haut-droit, bas-droit, bas-gauche
        s   = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()
        rect = np.zeros((4, 2), dtype=np.float32)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Dimensions cible (ratio billets FCFA ≈ 2:1)
        wA = np.linalg.norm(rect[2] - rect[3])
        wB = np.linalg.norm(rect[1] - rect[0])
        hA = np.linalg.norm(rect[1] - rect[2])
        hB = np.linalg.norm(rect[0] - rect[3])
        out_w = int(max(wA, wB))
        out_h = int(max(hA, hB))

        if out_w < 60 or out_h < 30:
            return pil_img, False

        dst = np.array([[0, 0], [out_w - 1, 0],
                         [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        M   = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(bgr, M, (out_w, out_h))
        result = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        return result, True

    # Fallback : bounding rect du plus grand contour
    cnt = cnts[0]
    area = cv2.contourArea(cnt)
    if area < img_area * 0.10:
        return pil_img, False

    x, y, cw, ch = cv2.boundingRect(cnt)
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + cw + mx)
    y2 = min(h, y + ch + my)
    cropped = bgr[y1:y2, x1:x2]
    result = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    return result, True


# ═══════════════════════════════════════════════════════════════════════════════
#  QUALITÉ PHOTO
# ═══════════════════════════════════════════════════════════════════════════════
def assess_photo_quality(pil_img: Image.Image) -> dict:
    """Évalue la qualité de la photo : flou, luminosité, taille."""
    img_np = np.array(pil_img.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w   = gray.shape

    # Flou (variance du Laplacien — plus c'est haut, plus c'est net)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = min(100.0, laplacian_var / 5.0)  # normalise ~500 → 100%

    # Luminosité
    mean_bright = float(gray.mean())
    if mean_bright < 40:
        brightness_score = mean_bright / 40.0 * 50
        brightness_label = "Trop sombre"
    elif mean_bright > 220:
        brightness_score = max(0, (255 - mean_bright) / 35.0 * 50)
        brightness_label = "Surexposée"
    else:
        # optimal ~100-170
        dist = abs(mean_bright - 140) / 80.0
        brightness_score = max(50, 100 - dist * 50)
        brightness_label = "Bonne"

    # Résolution
    pixels = h * w
    res_score = min(100.0, pixels / (640 * 480) * 100)

    overall = (sharpness * 0.5 + brightness_score * 0.3 + res_score * 0.2)
    overall = float(np.clip(overall, 0, 100))

    return {
        "overall": overall,
        "sharpness": sharpness,
        "brightness": brightness_score,
        "brightness_label": brightness_label,
        "resolution": res_score,
        "width": w,
        "height": h,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTRACTION DE FEATURES (2048 dims) — IDENTIQUE AU MODÈLE ENTRAÎNÉ
# ═══════════════════════════════════════════════════════════════════════════════
def extract_features(img: Image.Image) -> np.ndarray:
    """
    Extrait 2048 features d'une image PIL.
    Pipeline : redim 224×224 → histogramme RGB (768) + stats blocs (1280)
    ⚠️  Ne pas modifier sans ré-entraîner le modèle.
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

    # 2. Stats par blocs 16×16 → tronqué à 1280
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
    block_stats = (block_stats[:1280] if len(block_stats) >= 1280
                   else np.pad(block_stats, (0, 1280 - len(block_stats))))

    features.extend(block_stats.tolist())
    feat = np.array(features, dtype=np.float32)

    if len(feat) > 2048:
        feat = feat[:2048]
    elif len(feat) < 2048:
        feat = np.pad(feat, (0, 2048 - len(feat)))

    return feat.reshape(1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DU MODÈLE
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = joblib.load("modele.pkl")
        return model, None
    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════════════════════════════
#  PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════════
def predict_banknote(model, features: np.ndarray):
    """
    OneClassSVM : +1 = inlier (vrai billet), -1 = outlier (suspect)
    """
    prediction = int(model.predict(features)[0])
    score = float(model.decision_function(features)[0])

    score_clipped = np.clip(score, -3.0, 3.0)
    confidence = float((score_clipped + 3.0) / 6.0 * 100.0)

    if prediction == 1:
        status = "authentic" if confidence >= 60 else "uncertain"
    else:
        status = "fake"

    return status, confidence, score


# ═══════════════════════════════════════════════════════════════════════════════
#  RENDU RÉSULTAT
# ═══════════════════════════════════════════════════════════════════════════════
def render_result(status: str, confidence: float, score: float):
    if status == "authentic":
        col = "c-green"
        st.markdown("""
        <div class="res-card res-ok">
            <span class="res-emoji">✅</span>
            <p class="res-title">BILLET AUTHENTIQUE</p>
            <p class="res-msg">Ce billet semble <strong>vrai</strong>.<br>
            Vous pouvez l'accepter.</p>
        </div>""", unsafe_allow_html=True)
        bar_cls, decision = "bar-fill bar-green", "✓ ACCEPTER"

    elif status == "fake":
        col = "c-red"
        st.markdown("""
        <div class="res-card res-bad">
            <span class="res-emoji">🚫</span>
            <p class="res-title">FAUX BILLET !</p>
            <p class="res-msg">Ce billet présente des anomalies.<br>
            <strong>Refusez-le immédiatement.</strong></p>
        </div>""", unsafe_allow_html=True)
        bar_cls, decision = "bar-fill bar-red", "✗ REFUSER"

    else:
        col = "c-orng"
        st.markdown("""
        <div class="res-card res-unk">
            <span class="res-emoji">⚠️</span>
            <p class="res-title">RÉSULTAT INCERTAIN</p>
            <p class="res-msg">Photo insuffisante.<br>
            Reprenez en meilleure lumière.</p>
        </div>""", unsafe_allow_html=True)
        bar_cls, decision = "bar-fill bar-orange", "? VÉRIFIER"

    # Métriques
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-box">
            <div class="metric-val {col}">{confidence:.0f}%</div>
            <div class="metric-lbl">Confiance</div>
        </div>
        <div class="metric-box">
            <div class="metric-val {col}">{score:+.2f}</div>
            <div class="metric-lbl">Score IA</div>
        </div>
        <div class="metric-box">
            <div class="metric-val {col}">{decision}</div>
            <div class="metric-lbl">Décision</div>
        </div>
    </div>
    <div class="quality-bar">
        <div class="q-label"><span>Niveau de certitude</span><span>{confidence:.1f}%</span></div>
        <div class="bar-bg"><div class="{bar_cls}" style="width:{confidence:.1f}%"></div></div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  BARRE DE QUALITÉ PHOTO
# ═══════════════════════════════════════════════════════════════════════════════
def render_quality(q: dict):
    score = q["overall"]
    if score >= 65:
        cls, label = "bar-fill bar-green", f"Bonne qualité ({score:.0f}%)"
    elif score >= 35:
        cls, label = "bar-fill bar-orange", f"Qualité moyenne ({score:.0f}%)"
    else:
        cls, label = "bar-fill bar-red", f"Mauvaise qualité ({score:.0f}%)"

    st.markdown(f"""
    <div class="quality-bar">
        <div class="q-label"><span>📷 Qualité photo</span><span>{label}</span></div>
        <div class="bar-bg"><div class="{cls}" style="width:{score:.0f}%"></div></div>
    </div>""", unsafe_allow_html=True)

    if score < 35:
        st.markdown("""
        <div class="tip tip-warn">
        ⚠️ Photo de mauvaise qualité — résultat peu fiable.<br>
        Ajoutez de la lumière · Cadrez bien le billet · Nettoyez l'objectif
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():

    # ── En-tête ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <span class="app-logo">💴</span>
        <p class="app-title">BillGuard</p>
        <p class="app-tagline">DÉTECTEUR · BILLETS FCFA · GRATUIT · HORS LIGNE</p>
    </div>""", unsafe_allow_html=True)

    # ── Chargement modèle ────────────────────────────────────────────────────
    model, error = load_model()
    if model is None:
        st.error(f"⚠️ Modèle introuvable : {error}")
        st.info("Placez **modele.pkl** dans le même dossier que app.py.")
        st.stop()

    # ── Instructions visuelles (icônes = pas besoin de lire) ─────────────────
    st.markdown("""
    <div class="how-to">
        <div class="how-step">
            <span class="how-icon">📄</span>
            <div class="how-txt">Posez le billet sur table</div>
        </div>
        <div class="how-step">
            <span class="how-icon">📸</span>
            <div class="how-txt">Photographiez de près</div>
        </div>
        <div class="how-step">
            <span class="how-icon">🔍</span>
            <div class="how-txt">Attendez le résultat</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────────────────────────
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "📤  Choisir une photo du billet",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="visible",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Traitement ───────────────────────────────────────────────────────────
    if uploaded is not None:
        original = Image.open(uploaded)

        # Qualité photo
        quality = assess_photo_quality(original)
        render_quality(quality)

        # Recadrage automatique
        cropped, was_cropped = auto_crop_banknote(original)

        # Affichage côte à côte si recadrage
        if was_cropped:
            st.markdown("""<div class="crop-badge">✂️ Billet détecté et recadré automatiquement</div>""",
                        unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(original, caption="Photo originale", use_container_width=True)
            with col_b:
                st.image(cropped,  caption="✅ Billet recadré", use_container_width=True)
        else:
            st.markdown("""<div class="crop-badge crop-badge-warn">⚠️ Recadrage auto non détecté — vérifiez que le billet remplit la photo</div>""",
                        unsafe_allow_html=True)
            st.image(original, caption="Photo à analyser", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Bouton analyse
        if st.button("🔍   ANALYSER CE BILLET"):
            with st.spinner("⏳  Analyse en cours…"):
                try:
                    # Analyser l'image recadrée (ou originale si non détecté)
                    features = extract_features(cropped)
                    status, confidence, score = predict_banknote(model, features)
                    render_result(status, confidence, score)

                    # Conseils contextuels
                    if status == "fake":
                        st.markdown("""
                        <div class="tip tip-danger">
                        🚨 <strong>Que faire ?</strong><br>
                        • Ne remettez pas ce billet en circulation — c'est illégal<br>
                        • Signalez-le à la gendarmerie ou à la police<br>
                        • Contactez la BCEAO ou votre banque<br>
                        • Conservez-le comme preuve
                        </div>""", unsafe_allow_html=True)

                    elif status == "uncertain":
                        st.markdown("""
                        <div class="tip tip-warn">
                        📸 <strong>Améliorez la photo :</strong><br>
                        • Posez le billet à plat sur fond sombre<br>
                        • Éclairez bien (lumière du jour ou lampe)<br>
                        • Le billet doit remplir presque toute l'image<br>
                        • Nettoyez l'objectif de votre téléphone
                        </div>""", unsafe_allow_html=True)

                    else:
                        st.markdown("""
                        <div class="tip">
                        ✅ Vérifiez aussi manuellement :<br>
                        Filigrane · Bande holographique · Numéro de série en relief
                        </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Erreur d'analyse : {e}")
                    st.info("Essayez avec une photo plus nette et bien éclairée.")

    # ── À propos ─────────────────────────────────────────────────────────────
    with st.expander("ℹ️  À propos de BillGuard"):
        st.markdown("""
        **BillGuard** analyse vos billets FCFA par intelligence artificielle.
        Le modèle a appris à reconnaître les vrais billets : tout écart est signalé comme suspect.

        - 🔒 **Aucune photo envoyée** — analyse 100 % locale, sur votre appareil
        - 📵 **Fonctionne sans internet** — idéal pour le secteur informel
        - ⚡ **Résultat en quelques secondes**
        - ✂️ **Recadrage automatique** — le billet est isolé avant l'analyse

        > ⚠️ Outil d'aide à la décision. En cas de doute sérieux, consultez les autorités compétentes.
        """)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        💴 BillGuard · Secteur Informel · IA 100% Locale<br>
        Aucune donnée envoyée · Gratuit · Conçu pour tous
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
