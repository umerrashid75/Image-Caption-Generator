import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import os
from model_architecture import Seq2Seq
import nltk
from collections import Counter
import torchvision.models as models
import torch.nn as nn
import hashlib

# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG — wide layout, no scrolling                    ║
# ╚══════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="Neural Storyteller",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ╔══════════════════════════════════════════════════════════════╗
# ║  PREMIUM DARK SAAS UI — CSS + PARTICLES + GLASSMORPHISM    ║
# ╚══════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
/* ─────────────────── FONTS ─────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─────────────────── NO-SCROLL VIEWPORT ─────────────────── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    overflow: hidden !important;
    height: 100vh !important;
    max-height: 100vh !important;
}

/* ─────────────────── ANIMATED GRADIENT BACKGROUND ─────────────────── */
.stApp {
    background: #06070b !important;
    position: relative !important;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 80%, rgba(139,92,246,0.12), transparent 60%),
        radial-gradient(ellipse 70% 50% at 80% 20%, rgba(236,72,153,0.08), transparent 60%),
        radial-gradient(ellipse 60% 40% at 50% 50%, rgba(59,130,246,0.06), transparent 60%);
    animation: ns-bg-shift 12s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes ns-bg-shift {
    0% { filter: hue-rotate(0deg) brightness(1); }
    50% { filter: hue-rotate(15deg) brightness(1.1); }
    100% { filter: hue-rotate(-10deg) brightness(1); }
}

/* ─────────────────── BASE TYPOGRAPHY ─────────────────── */
html, body, [class*="css"], p, span, label, li,
.stMarkdown, .stMarkdown p {
    font-family: 'Inter', sans-serif !important;
    color: #b0b7c3 !important;
}
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f0f0f0 !important;
}

/* ─────────────────── HIDE CHROME ─────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1rem 2.5rem 0.5rem !important;
    max-width: 100% !important;
}

hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.25), transparent) !important;
    margin: 0.6rem 0 !important;
}

/* ─────────────────── CENTERED HEADER ─────────────────── */
.ns-header {
    text-align: center;
    padding: 0.8rem 0 0.4rem;
    position: relative;
    z-index: 2;
}
.ns-header-glow {
    position: absolute;
    top: -30px; left: 50%;
    transform: translateX(-50%);
    width: 350px; height: 100px;
    background: radial-gradient(circle, rgba(139,92,246,0.2), rgba(236,72,153,0.08) 50%, transparent 70%);
    pointer-events: none;
    filter: blur(35px);
    animation: ns-glow-breathe 5s ease-in-out infinite;
}
@keyframes ns-glow-breathe {
    0%, 100% { opacity: 0.6; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.15); }
}
.ns-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(139,92,246,0.1);
    border: 1px solid rgba(139,92,246,0.25);
    color: #a78bfa !important;
    font-size: 0.58rem; font-weight: 600;
    letter-spacing: 2.5px; text-transform: uppercase;
    padding: 4px 16px; border-radius: 100px;
    margin-bottom: 0.6rem;
    font-family: 'JetBrains Mono', monospace !important;
}
.ns-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.6rem; font-weight: 800;
    letter-spacing: -2px; line-height: 1.05;
    margin: 0 0 0.3rem;
    background: linear-gradient(135deg, #a78bfa, #c084fc, #e879f9, #f472b6, #a78bfa);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: ns-title-shimmer 6s ease-in-out infinite;
}
@keyframes ns-title-shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 300% center; }
}
.ns-tagline {
    color: #6b7280 !important; font-size: 0.85rem;
    font-weight: 400; margin: 0; letter-spacing: 0.3px;
}

/* ─────────────────── GLASS CARD ─────────────────── */
.ns-glass {
    background: rgba(15,17,25,0.55);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(139,92,246,0.1);
    border-radius: 20px;
    padding: 1.4rem 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 8px 32px rgba(0,0,0,0.3),
        inset 0 1px 0 rgba(255,255,255,0.04);
    z-index: 2;
}
/* Blurred radial glow behind cards */
.ns-glass::before {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 140%; height: 140%;
    background: radial-gradient(circle, rgba(139,92,246,0.06), transparent 60%);
    pointer-events: none;
    z-index: -1;
}

.ns-glass-label {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: #6b7280 !important;
    margin: 0 0 0.8rem;
    font-family: 'JetBrains Mono', monospace !important;
}
.ns-glass-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(139,92,246,0.2), transparent);
}

/* ─────────────────── FILE UPLOAD ─────────────────── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
.ns-upload-zone {
    border: 1.5px dashed rgba(139,92,246,0.25);
    border-radius: 14px;
    padding: 2px;
    transition: all 0.4s ease;
    position: relative;
    overflow: hidden;
}
.ns-upload-zone:hover {
    border-color: rgba(139,92,246,0.5);
    box-shadow: 0 0 30px rgba(139,92,246,0.06);
}

/* ─────────────────── IMAGE PREVIEW ─────────────────── */
.stImage {
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.4), 0 0 0 1px rgba(139,92,246,0.06) !important;
}
.stImage img { border-radius: 14px !important; }

/* ─────────────────── CAPTION CARD ─────────────────── */
.ns-caption-card {
    background: rgba(15,17,25,0.65);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(139,92,246,0.12);
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03);
    animation: ns-card-fade 0.6s cubic-bezier(.4,0,.2,1);
    position: relative;
}
.ns-caption-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #8b5cf6, #ec4899, #8b5cf6);
    background-size: 200% auto;
    animation: ns-bar-slide 3s linear infinite;
}
@keyframes ns-bar-slide { 0%{background-position:0%} 100%{background-position:200%} }
@keyframes ns-card-fade {
    from { opacity:0; transform: translateY(14px) scale(.97); }
    to   { opacity:1; transform: translateY(0) scale(1); }
}

.ns-cap-top {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 18px;
    background: rgba(255,255,255,0.015);
    border-bottom: 1px solid rgba(139,92,246,0.08);
}
.ns-cap-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: linear-gradient(135deg, #8b5cf6, #ec4899);
    box-shadow: 0 0 10px rgba(139,92,246,0.5);
    animation: ns-dot-pulse 2.5s ease-in-out infinite;
}
@keyframes ns-dot-pulse {
    0%,100% { opacity:1; box-shadow:0 0 10px rgba(139,92,246,.5) }
    50% { opacity:.4; box-shadow:0 0 4px rgba(139,92,246,.2) }
}
.ns-cap-tag {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.58rem; font-weight: 500;
    color: #6b7280 !important; letter-spacing: 1.5px; text-transform: uppercase;
}
.ns-cap-method {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.55rem; font-weight: 600;
    color: #a78bfa !important;
    background: rgba(139,92,246,0.1);
    padding: 3px 10px; border-radius: 6px;
    border: 1px solid rgba(139,92,246,0.2);
    letter-spacing: 0.8px;
}
.ns-cap-body { padding: 18px 22px 22px; }

/* ─── typing animation ─── */
.ns-cap-text {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.15rem; font-weight: 500;
    color: #e2e5ea !important;
    line-height: 1.75; margin: 0; letter-spacing: 0.2px;
    overflow: hidden;
    border-right: 2px solid #8b5cf6;
    white-space: nowrap;
    animation: ns-typing 2.5s steps(60,end) forwards, ns-blink .75s step-end infinite;
    max-width: 100%;
}
@keyframes ns-typing { from{width:0} to{width:100%} }
@keyframes ns-blink { 50%{border-color:transparent} }

/* Long captions: wrap after typing finishes */
.ns-cap-text.done {
    white-space: normal;
    border-right: none;
    animation: none;
}

/* ─── quote marks ─── */
.ns-cap-text::before {
    content: '\\201C'; font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg,#8b5cf6,#ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 0; vertical-align: -0.35em; margin-right: 4px;
}
.ns-cap-text::after {
    content: '\\201D'; font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg,#ec4899,#8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 0; vertical-align: -0.35em; margin-left: 4px;
}

/* Greedy variant accent */
.ns-caption-card.greedy::before {
    background: linear-gradient(90deg,#3b82f6,#06b6d4,#3b82f6) !important;
    background-size: 200% auto !important;
}
.ns-caption-card.greedy .ns-cap-dot {
    background: linear-gradient(135deg,#3b82f6,#06b6d4) !important;
    box-shadow: 0 0 10px rgba(59,130,246,.5) !important;
}
.ns-caption-card.greedy .ns-cap-method {
    color: #60a5fa !important; background: rgba(59,130,246,.1) !important;
    border-color: rgba(59,130,246,.2) !important;
}

/* ─────────────────── GRADIENT GLOW BUTTON ─────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 40%, #ec4899 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 600 !important; font-size: 0.82rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.35s cubic-bezier(.4,0,.2,1) !important;
    box-shadow: 0 4px 20px rgba(139,92,246,0.3), 0 0 40px rgba(139,92,246,0.08) !important;
    position: relative !important; overflow: hidden !important;
}
.stButton > button::after {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, transparent 30%, rgba(255,255,255,0.15) 50%, transparent 70%);
    background-size: 200% auto;
    animation: ns-btn-shine 3s ease-in-out infinite;
}
@keyframes ns-btn-shine { 0%{background-position:200%} 100%{background-position:-200%} }
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 32px rgba(139,92,246,0.45), 0 0 60px rgba(139,92,246,0.12) !important;
}
.stButton > button:active { transform: translateY(0) scale(1) !important; }

/* ─────────────────── PROGRESS ─────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #8b5cf6, #c084fc, #e879f9, #8b5cf6) !important;
    background-size: 300% auto !important;
    animation: ns-prog 2s ease infinite !important; border-radius: 6px !important;
}
@keyframes ns-prog { 0%{background-position:0%} 100%{background-position:300%} }
.stProgress > div > div {
    background: rgba(139,92,246,.08) !important; border-radius: 6px !important;
}

/* ─────────────────── HIDE UPLOAD ZONE AFTER FILE SELECTED ─────────────────── */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    display: none !important;
}

/* ─────────────────── EMPTY STATE ─────────────────── */
.ns-empty {
    text-align: center; padding: 3rem 1rem 1rem;
}
.ns-empty-orb {
    width: 80px; height: 80px; margin: 0 auto 1rem;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, rgba(139,92,246,.2), rgba(236,72,153,.1) 60%, transparent 70%);
    display: flex; align-items: center; justify-content: center;
    font-size: 2.2rem;
    animation: ns-orb-float 5s ease-in-out infinite;
    box-shadow: 0 0 50px rgba(139,92,246,.1);
    border: 1px solid rgba(139,92,246,.1);
}
@keyframes ns-orb-float {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}
.ns-empty-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.1rem; font-weight: 600; color: #d1d5db !important;
    margin: 0 0 0.3rem;
}
.ns-empty-sub {
    color: #4b5563 !important; font-size: 0.82rem;
    line-height: 1.6; margin: 0;
}
.ns-empty-sub strong { color: #6b7280 !important; }

/* ─────────────────── FOOTER ─────────────────── */
.ns-footer {
    text-align: center; padding: 0.4rem 0;
    font-size: 0.6rem; color: #2d3348 !important;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace !important;
    position: relative; z-index: 2;
}
.ns-footer span {
    background: linear-gradient(135deg,#8b5cf6,#ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* ─────────────────── SCROLLBAR HIDE ─────────────────── */
::-webkit-scrollbar { display: none; }
[data-testid="column"] { background: transparent !important; }
.stAlert {
    background: rgba(15,17,25,.8) !important;
    border: 1px solid rgba(139,92,246,.1) !important;
    border-radius: 12px !important;
}
</style>

<!-- ─────────── PARTICLE CANVAS ─────────── -->
<canvas id="ns-particles" style="
    position:fixed; inset:0; z-index:0;
    pointer-events:none; width:100vw; height:100vh;
"></canvas>
<script>
(function(){
    const c = document.getElementById('ns-particles');
    if (!c) return;
    const ctx = c.getContext('2d');
    let W, H;
    function resize(){ W=c.width=window.innerWidth; H=c.height=window.innerHeight; }
    resize(); window.addEventListener('resize', resize);

    const N = 45;
    const dots = Array.from({length:N}, ()=>({
        x: Math.random()*W, y: Math.random()*H,
        vx: (Math.random()-.5)*.25, vy: (Math.random()-.5)*.25,
        r: Math.random()*1.8+.6,
        a: Math.random()*.25+.08
    }));

    function draw(){
        ctx.clearRect(0,0,W,H);
        dots.forEach(d=>{
            d.x+=d.vx; d.y+=d.vy;
            if(d.x<0)d.x=W; if(d.x>W)d.x=0;
            if(d.y<0)d.y=H; if(d.y>H)d.y=0;
            ctx.beginPath();
            ctx.arc(d.x,d.y,d.r,0,Math.PI*2);
            ctx.fillStyle='rgba(139,92,246,'+d.a+')';
            ctx.fill();
        });
        // Draw faint lines between close dots
        for(let i=0;i<N;i++){
            for(let j=i+1;j<N;j++){
                const dx=dots[i].x-dots[j].x, dy=dots[i].y-dots[j].y;
                const dist=Math.sqrt(dx*dx+dy*dy);
                if(dist<150){
                    ctx.beginPath();
                    ctx.moveTo(dots[i].x,dots[i].y);
                    ctx.lineTo(dots[j].x,dots[j].y);
                    ctx.strokeStyle='rgba(139,92,246,'+(0.04*(1-dist/150))+')';
                    ctx.lineWidth=0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }
    draw();
})();
</script>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  NLTK DATA                                                  ║
# ╚══════════════════════════════════════════════════════════════╝
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ╔══════════════════════════════════════════════════════════════╗
# ║  VOCABULARY CLASS                                           ║
# ╚══════════════════════════════════════════════════════════════╝
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

# ╔══════════════════════════════════════════════════════════════╗
# ║  MODEL CONFIG & LOADING                                     ║
# ╚══════════════════════════════════════════════════════════════╝
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1  # Must match the model trained with 1 LSTM layer
MODEL_PATH = "neural_storyteller.pth"
VOCAB_PATH = "vocab.pkl"

@st.cache_resource
def load_model_and_vocab():
    if not os.path.exists(VOCAB_PATH):
        st.error(f"Vocabulary file not found at `{VOCAB_PATH}`.")
        return None, None, None

    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    modules = list(resnet.children())[:-1]
    resnet = nn.Sequential(*modules)
    resnet.to(DEVICE)
    resnet.eval()

    model = Seq2Seq(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at `{MODEL_PATH}`.")
        return None, None, vocab

    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        return None, None, vocab

    return resnet, model, vocab

resnet, model, vocab = load_model_and_vocab()

# ╔══════════════════════════════════════════════════════════════╗
# ║  IMAGE TRANSFORM                                            ║
# ╚══════════════════════════════════════════════════════════════╝
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# ╔══════════════════════════════════════════════════════════════╗
# ║  CAPTION GENERATION                                         ║
# ╚══════════════════════════════════════════════════════════════╝
def generate_caption(image, resnet, model, vocab, method='greedy'):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = resnet(image_tensor)
        features = features.reshape(features.size(0), -1)
        if method == 'beam':
            caption = model.caption_image_beam_search(features, vocab)
        else:
            caption = model.caption_image(features, vocab)
    return " ".join(caption)

# ╔══════════════════════════════════════════════════════════════╗
# ║  SESSION STATE                                              ║
# ╚══════════════════════════════════════════════════════════════╝
if "last_hash" not in st.session_state:
    st.session_state.last_hash = None
if "beam_cap" not in st.session_state:
    st.session_state.beam_cap = None
if "greedy_cap" not in st.session_state:
    st.session_state.greedy_cap = None

def get_hash(f):
    return hashlib.md5(f.getvalue()).hexdigest()


# ╔══════════════════════════════════════════════════════════════╗
# ║  ██  L A Y O U T  ██                                       ║
# ╚══════════════════════════════════════════════════════════════╝

# ── CENTERED HEADER ──
st.markdown("""
<div class="ns-header">
    <div class="ns-header-glow"></div>
    <div class="ns-badge">✦ AI-Powered Image Captioning</div>
    <p class="ns-title">Neural Storyteller</p>
    <p class="ns-tagline">Upload an image — our Seq2Seq model writes the story.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── TWO-COLUMN LAYOUT ──
col_left, col_spacer, col_right = st.columns([5, 0.4, 5])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LEFT COLUMN — Upload & Preview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with col_left:
    st.markdown('<div class="ns-glass">', unsafe_allow_html=True)
    st.markdown('<div class="ns-glass-label">📤 Image Input</div>', unsafe_allow_html=True)

    st.markdown('<div class="ns-upload-zone">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop an image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        current_hash = get_hash(uploaded_file)

        # Detect new image → clear old captions
        if current_hash != st.session_state.last_hash:
            st.session_state.last_hash = current_hash
            st.session_state.beam_cap = None
            st.session_state.greedy_cap = None

        st.markdown('<div class="ns-glass-label" style="margin-top:0.8rem;">🖼️ Preview</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    else:
        st.markdown("""
        <div class="ns-empty">
            <div class="ns-empty-orb">📸</div>
            <p class="ns-empty-title">No image yet</p>
            <p class="ns-empty-sub">
                Drop a <strong>.jpg</strong> or <strong>.png</strong> above<br/>
                to see the magic happen.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RIGHT COLUMN — Caption Output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with col_right:
    st.markdown('<div class="ns-glass">', unsafe_allow_html=True)
    st.markdown('<div class="ns-glass-label">✨ AI Output</div>', unsafe_allow_html=True)

    if uploaded_file is not None and model and vocab and resnet:
        # Auto-generate beam caption for new images
        if st.session_state.beam_cap is None:
            bar = st.progress(0)
            for i in range(100):
                bar.progress(i + 1)
            st.session_state.beam_cap = generate_caption(image, resnet, model, vocab, method='beam')
            bar.empty()

        beam_caption = st.session_state.beam_cap

        # Display beam search caption card with typing animation
        st.markdown(f"""
        <div class="ns-caption-card">
            <div class="ns-cap-top">
                <div class="ns-cap-dot"></div>
                <span class="ns-cap-tag">Generated Caption</span>
                <span class="ns-cap-method">✦ BEAM SEARCH</span>
            </div>
            <div class="ns-cap-body">
                <p class="ns-cap-text">{beam_caption}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Typing animation auto-stop script
        st.markdown("""
        <script>
        setTimeout(function(){
            document.querySelectorAll('.ns-cap-text').forEach(function(el){
                el.classList.add('done');
            });
        }, 2600);
        </script>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="ns-empty">
            <div class="ns-empty-orb">🧠</div>
            <p class="ns-empty-title">Awaiting input…</p>
            <p class="ns-empty-sub">
                Upload an image on the left and<br/>
                the AI caption will appear here.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("---")
st.markdown("""
<div class="ns-footer">
    BUILT WITH <span>♥</span> USING PYTORCH &middot; RESNET-50 &middot; LSTM &nbsp;|&nbsp; GENAI ASSIGNMENT
</div>
""", unsafe_allow_html=True)