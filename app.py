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

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Neural Storyteller",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════
#  STYLED DARK UI — CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── No-scroll viewport ── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    overflow: hidden !important;
    height: 100vh !important;
    max-height: 100vh !important;
}

/* ── Dark background with subtle warmth ── */
.stApp {
    background: #09090b !important;
}

/* ── Typography ── */
html, body, [class*="css"], p, span, label, li,
.stMarkdown, .stMarkdown p {
    font-family: 'Inter', sans-serif !important;
    color: #a1a1aa !important;
}
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #fafafa !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.8rem 3rem 1rem !important;
    max-width: 100% !important;
}

/* ── Header ── */
.ns-header {
    text-align: center;
    padding: 0 0 1rem;
}
.ns-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.55rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #8b5cf6 !important;
    background: rgba(139, 92, 246, 0.08);
    border: 1px solid rgba(139, 92, 246, 0.15);
    padding: 4px 14px;
    border-radius: 100px;
    margin-bottom: 0.6rem;
}
.ns-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.4rem;
    font-weight: 800;
    color: #fafafa !important;
    margin: 0;
    letter-spacing: -1.5px;
}
.ns-title span {
    color: #8b5cf6 !important;
}
.ns-subtitle {
    font-size: 0.88rem;
    color: #52525b !important;
    margin: 0.3rem 0 0;
    font-weight: 400;
}

/* ── Divider ── */
.ns-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #27272a, transparent);
    margin: 0.2rem 0 1.2rem;
    border: none;
}

/* ── Section labels ── */
.ns-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3f3f46 !important;
    margin: 0 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.ns-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1c1c1e;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(24, 24, 27, 0.5) !important;
    border: 1.5px dashed #27272a !important;
    border-radius: 14px !important;
    padding: 2.5rem 1rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #8b5cf6 !important;
    background: rgba(139, 92, 246, 0.03) !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] small {
    color: #3f3f46 !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button {
    background: transparent !important;
    border: 1px solid #27272a !important;
    color: #71717a !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button:hover {
    border-color: #8b5cf6 !important;
    color: #a78bfa !important;
}
/* Hide dropzone once file selected */
[data-testid="stFileUploader"]:has([data-testid="stFileUploaderFile"]) [data-testid="stFileUploaderDropzone"] {
    display: none !important;
}
[data-testid="stFileUploaderFile"] {
    display: none !important;
}

/* ── Image preview ── */
.stImage {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #1c1c1e !important;
}
.stImage img { border-radius: 14px !important; }

/* ── Caption card ── */
.ns-caption-card {
    background: rgba(24, 24, 27, 0.4);
    border: 1px solid #27272a;
    border-radius: 16px;
    overflow: hidden;
    animation: ns-fade 0.6s ease-out;
}
@keyframes ns-fade {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.ns-cap-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    border-bottom: 1px solid #1c1c1e;
}
.ns-cap-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #8b5cf6;
}
.ns-cap-badge {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.55rem;
    font-weight: 500;
    color: #52525b !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.ns-cap-body {
    padding: 20px 22px;
}
.ns-cap-text {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.2rem;
    font-weight: 500;
    color: #e4e4e7 !important;
    line-height: 1.8;
    margin: 0;
}
.ns-cap-text::before {
    content: '"';
    color: #8b5cf6 !important;
    font-weight: 700;
    font-size: 1.5rem;
    margin-right: 2px;
}
.ns-cap-text::after {
    content: '"';
    color: #8b5cf6 !important;
    font-weight: 700;
    font-size: 1.5rem;
    margin-left: 2px;
}

/* ── Empty state ── */
.ns-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 300px;
}
.ns-empty-icon {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
    opacity: 0.3;
}
.ns-empty-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem;
    font-weight: 600;
    color: #3f3f46 !important;
    margin: 0 0 0.2rem;
}
.ns-empty-sub {
    color: #27272a !important;
    font-size: 0.8rem;
    margin: 0;
}

/* ── Progress ── */
.stProgress > div > div > div {
    background: #8b5cf6 !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: #18181b !important;
    border-radius: 4px !important;
}

/* ── Footer ── */
.ns-footer {
    text-align: center;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.58rem;
    color: #27272a !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 0.8rem 0 0;
}
.ns-footer span { color: #8b5cf6 !important; }

/* ── Misc ── */
[data-testid="column"] { background: transparent !important; }
::-webkit-scrollbar { display: none; }
.stAlert {
    background: #0a0a0a !important;
    border: 1px solid #27272a !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  NLTK
# ══════════════════════════════════════════════════════════════
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ══════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
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

# ══════════════════════════════════════════════════════════════
#  IMAGE TRANSFORM
# ══════════════════════════════════════════════════════════════
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# ══════════════════════════════════════════════════════════════
#  CAPTION GENERATION
# ══════════════════════════════════════════════════════════════
def generate_caption(image, resnet, model, vocab, method='beam'):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = resnet(image_tensor)
        features = features.reshape(features.size(0), -1)
        if method == 'beam':
            caption = model.caption_image_beam_search(features, vocab)
        else:
            caption = model.caption_image(features, vocab)
    return " ".join(caption)

# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
if "last_hash" not in st.session_state:
    st.session_state.last_hash = None
if "caption" not in st.session_state:
    st.session_state.caption = None

def get_hash(f):
    return hashlib.md5(f.getvalue()).hexdigest()

# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════

# ── Header ──
st.markdown("""
<div class="ns-header">
    <div class="ns-badge">✦ GenAI Project</div>
    <p class="ns-title">Neural <span>Storyteller</span></p>
    <p class="ns-subtitle">Upload an image and let the model write its story</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="ns-divider"></div>', unsafe_allow_html=True)

# ── Two columns ──
col_left, col_gap, col_right = st.columns([5, 0.3, 5])

# ── LEFT: Upload & Preview ──
with col_left:
    st.markdown('<div class="ns-label">📷 Image Input</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        current_hash = get_hash(uploaded_file)

        # New image → clear old caption
        if current_hash != st.session_state.last_hash:
            st.session_state.last_hash = current_hash
            st.session_state.caption = None

        st.image(image, use_column_width=True)

    else:
        st.markdown("""
        <div class="ns-empty">
            <div class="ns-empty-icon">📸</div>
            <p class="ns-empty-title">No image uploaded</p>
            <p class="ns-empty-sub">Drop a .jpg or .png above</p>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT: Caption Output ──
with col_right:
    st.markdown('<div class="ns-label">✨ Caption Output</div>', unsafe_allow_html=True)

    if uploaded_file is not None and model and vocab and resnet:
        # Generate caption if not cached
        if st.session_state.caption is None:
            bar = st.progress(0)
            for i in range(100):
                bar.progress(i + 1)
            st.session_state.caption = generate_caption(image, resnet, model, vocab, method='beam')
            bar.empty()

        st.markdown(f"""
        <div class="ns-caption-card">
            <div class="ns-cap-header">
                <div class="ns-cap-dot"></div>
                <span class="ns-cap-badge">Beam Search</span>
            </div>
            <div class="ns-cap-body">
                <p class="ns-cap-text">{st.session_state.caption}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="ns-empty">
            <div class="ns-empty-icon">🧠</div>
            <p class="ns-empty-title">Awaiting input</p>
            <p class="ns-empty-sub">Caption will appear here</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div class="ns-footer">
    Built with <span>♥</span> using PyTorch · ResNet-50 · LSTM &nbsp;|&nbsp; GenAI Assignment
</div>
""", unsafe_allow_html=True)