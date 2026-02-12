import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="BillGuard", page_icon="💵", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@900&display=swap');
* { font-family: 'Nunito', sans-serif !important; }
.stApp { background: #0f1923; }
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }

.logo { text-align:center; padding: 28px 0 8px; }
.logo-icon { font-size: 72px; }
.logo-name {
    font-size: 38px; font-weight: 900; letter-spacing: 3px; margin:0;
    background: linear-gradient(90deg, #00e5ff, #ffffff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.cam-wrap { display:flex; justify-content:center; padding: 18px 0 8px; }
.cam-btn {
    background: linear-gradient(135deg, #0055cc, #00aaff);
    border-radius: 32px; width: 230px; height: 230px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 8px;
    box-shadow: 0 0 60px rgba(0,170,255,0.5);
}
.cam-icon { font-size: 90px; line-height: 1; }
.cam-txt  { font-size: 22px; font-weight: 900; color: #fff; }

.result-ok  {
    background: #071f10; border: 5px solid #00e676;
    border-radius: 30px; padding: 36px 20px; text-align: center; margin:14px 0;
}
.result-bad {
    background: #200505; border: 5px solid #ff1744;
    border-radius: 30px; padding: 36px 20px; text-align: center; margin:14px 0;
}
.result-unk {
    background: #1f1500; border: 5px solid #ffd600;
    border-radius: 30px; padding: 36px 20px; text-align: center; margin:14px 0;
}
.r-icon { font-size: 96px; display:block; margin-bottom:8px; }
.r-ok  { font-size: 34px; font-weight:900; color:#00e676; margin:0; }
.r-bad { font-size: 34px; font-weight:900; color:#ff1744; margin:0; }
.r-unk { font-size: 34px; font-weight:900; color:#ffd600; margin:0; }

.bar-wrap { background:rgba(255,255,255,.1); border-radius:99px; height:20px; margin:14px 0 6px; overflow:hidden; }
.bar-g { height:100%; border-radius:99px; background:linear-gradient(90deg,#00c853,#00e676); }
.bar-r { height:100%; border-radius:99px; background:linear-gradient(90deg,#b71c1c,#ff1744); }
.bar-y { height:100%; border-radius:99px; background:linear-gradient(90deg,#f57f17,#ffd600); }
.bar-pct { font-size:24px; font-weight:900; color:#fff !important; }

.stButton > button {
    background: linear-gradient(135deg, #00aa44, #00e676) !important;
    color: #000 !important; border: none !important; border-radius: 22px !important;
    font-size: 24px !important; font-weight: 900 !important;
    padding: 20px !important; width: 100% !important;
    box-shadow: 0 6px 30px rgba(0,200,80,.5) !important;
}

[data-testid="stFileUploader"] { display:none !important; }
[data-testid="stImage"] img {
    border-radius: 20px !important;
    border: 3px solid rgba(255,255,255,.15) !important;
}
p, span, label, div { color: #b0cce0 !important; }
</style>
""", unsafe_allow_html=True)


def extract_features(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB').resize((128, 128))
    arr = np.array(img, dtype=np.float32) / 255.0
    feats = []
    for c in range(3):
        h, _ = np.histogram(arr[:,:,c], bins=32, range=(0,1))
        h = h.astype(np.float32)
        if h.sum() > 0: h /= h.sum()
        feats.extend(h.tolist())
    for c in range(3):
        ch = arr[:,:,c]
        feats += [ch.mean(), ch.std(), float(np.percentile(ch,25)), float(np.percentile(ch,75))]
    for c in range(3):
        ch = arr[:,:,c]
        for i in range(0, 128, 16):
            for j in range(0, 128, 16):
                blk = ch[i:i+16, j:j+16]
                feats += [float(blk.mean()), float(blk.std()), float(blk.max()-blk.min())]
    r = arr[:,:,0].mean()+1e-6; g = arr[:,:,1].mean()+1e-6; b = arr[:,:,2].mean()+1e-6
    feats += [g/r, b/r, b/g]
    return np.array(feats, dtype=np.float32)


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load("modele.pkl"), None
    except Exception as e:
        return None, str(e)


def predict(model, img: Image.Image):
    feat = extract_features(img).reshape(1, -1)
    pred  = int(model.predict(feat)[0])
    score = float(model.decision_function(feat)[0])
    conf  = float((np.clip(score, -3.0, 3.0) + 3.0) / 6.0 * 100.0)
    if pred == 1 and conf >= 55:
        return "ok", conf
    elif pred == 1:
        return "unk", conf
    else:
        return "bad", max(0.0, 100.0 - conf)


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="logo">
  <div class="logo-icon">💵</div>
  <p class="logo-name">BILLGUARD</p>
</div>
""", unsafe_allow_html=True)

model, err = load_model()
if model is None:
    st.error(f"Modèle manquant : {err}")
    st.stop()

# File uploader caché — déclenché par le gros bouton 📷
uploaded = st.file_uploader(
    "photo", type=["jpg","jpeg","png","webp"],
    label_visibility="collapsed"
)

if uploaded is None:
    # Afficher le gros bouton photo quand rien n'est chargé
    st.markdown("""
    <div class="cam-wrap">
      <div class="cam-btn">
        <span class="cam-icon">📷</span>
        <span class="cam-txt">PHOTO</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    image = Image.open(uploaded)
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.image(image, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍  ANALYSER"):
        with st.spinner(""):
            status, conf = predict(model, image)

        if status == "ok":
            st.markdown(f"""
            <div class="result-ok">
              <span class="r-icon">✅</span>
              <p class="r-ok">VRAI BILLET</p>
              <div class="bar-wrap"><div class="bar-g" style="width:{conf:.0f}%"></div></div>
              <span class="bar-pct">{conf:.0f}%</span>
            </div>""", unsafe_allow_html=True)

        elif status == "bad":
            st.markdown(f"""
            <div class="result-bad">
              <span class="r-icon">🚫</span>
              <p class="r-bad">FAUX BILLET</p>
              <div class="bar-wrap"><div class="bar-r" style="width:{conf:.0f}%"></div></div>
              <span class="bar-pct">{conf:.0f}%</span>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="result-unk">
              <span class="r-icon">⚠️</span>
              <p class="r-unk">REPRENDRE PHOTO</p>
              <div class="bar-wrap"><div class="bar-y" style="width:{conf:.0f}%"></div></div>
              <span class="bar-pct">{conf:.0f}%</span>
            </div>""", unsafe_allow_html=True)
