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
    page_title="Image Caption AI",
    page_icon="◻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════
#  MINIMAL BLACK & WHITE CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── No scroll, full viewport ── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    overflow: hidden !important;
    height: 100vh !important;
    max-height: 100vh !important;
}

/* ── Pure black base ── */
.stApp {
    background: #000000 !important;
}

/* ── White text everywhere ── */
html, body, [class*="css"], p, span, label, li,
.stMarkdown, .stMarkdown p, h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    color: #ffffff !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem 1rem !important;
    max-width: 100% !important;
}

/* ── Title ── */
.ns-title {
    text-align: center;
    font-family: 'Inter', sans-serif !important;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff !important;
    margin: 0;
    letter-spacing: -1px;
}
.ns-subtitle {
    text-align: center;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem;
    font-weight: 400;
    color: #666666 !important;
    margin: 0.3rem 0 1.5rem;
}

/* ── Panel (both columns) ── */
.ns-panel {
    padding: 0.5rem 0;
    display: flex;
    flex-direction: column;
}

/* ── Panel label ── */
.ns-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #555555 !important;
    margin: 0 0 1rem;
    font-family: 'Inter', sans-serif !important;
}

/* ── Upload placeholder (+ sign box) ── */
.ns-upload-placeholder {
    border: 1px dashed #333333;
    border-radius: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    min-height: 320px;
    cursor: pointer;
    transition: border-color 0.2s ease;
}
.ns-upload-placeholder:hover {
    border-color: #555555;
}
.ns-plus {
    font-size: 3rem;
    font-weight: 300;
    color: #333333 !important;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.ns-upload-hint {
    font-size: 0.8rem;
    color: #444444 !important;
    font-weight: 400;
}

/* ── Hide default file uploader styling ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] > div {
    padding: 0 !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: 1px dashed #333333 !important;
    border-radius: 16px !important;
    padding: 3rem 1rem !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #555555 !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button {
    color: #444444 !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button {
    background: transparent !important;
    border: 1px solid #333333 !important;
    color: #888888 !important;
    border-radius: 8px !important;
}

/* Hide dropzone after file selected */
[data-testid="stFileUploader"]:has([data-testid="stFileUploaderFile"]) [data-testid="stFileUploaderDropzone"] {
    display: none !important;
}
/* Hide file name chip */
[data-testid="stFileUploaderFile"] {
    display: none !important;
}

/* ── Image preview ── */
.stImage {
    border-radius: 12px !important;
    overflow: hidden !important;
}
.stImage img {
    border-radius: 12px !important;
}

/* ── Caption output ── */
.ns-caption-area {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    min-height: 320px;
}
.ns-caption-placeholder {
    font-size: 0.95rem;
    color: #333333 !important;
    font-weight: 400;
    font-style: italic;
}
.ns-caption-result {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.3rem;
    font-weight: 500;
    color: #ffffff !important;
    line-height: 1.8;
    text-align: center;
    max-width: 90%;
    animation: ns-fade-in 0.8s ease-out;
}
@keyframes ns-fade-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Generating indicator ── */
.ns-generating {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #555555 !important;
    font-size: 0.85rem;
}
.ns-dot-loader {
    display: inline-flex; gap: 4px;
}
.ns-dot-loader span {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #555555;
    animation: ns-dot-bounce 1.2s ease-in-out infinite;
}
.ns-dot-loader span:nth-child(2) { animation-delay: 0.15s; }
.ns-dot-loader span:nth-child(3) { animation-delay: 0.3s; }
@keyframes ns-dot-bounce {
    0%, 80%, 100% { opacity: 0.3; }
    40% { opacity: 1; }
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: #ffffff !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: #1a1a1a !important;
    border-radius: 4px !important;
}

/* ── Footer ── */
.ns-footer {
    text-align: center;
    font-size: 0.65rem;
    color: #2a2a2a !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 1rem;
    font-family: 'Inter', sans-serif !important;
}

/* ── Columns ── */
[data-testid="column"] { background: transparent !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { display: none; }

/* ── Alerts ── */
.stAlert {
    background: #0a0a0a !important;
    border: 1px solid #222222 !important;
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
st.markdown('<p class="ns-title">Image Caption AI</p>', unsafe_allow_html=True)
st.markdown('<p class="ns-subtitle">Upload an image to generate a caption</p>', unsafe_allow_html=True)

# ── Two columns ──
col_left, col_gap, col_right = st.columns([5, 0.3, 5])

# ── LEFT: Upload & Preview ──
with col_left:

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

        st.image(image, use_container_width=True)

    else:
        st.markdown("""
        <div class="ns-upload-placeholder">
            <span class="ns-plus">+</span>
            <span class="ns-upload-hint">Click to upload image</span>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT: Caption Output ──
with col_right:

    if uploaded_file is not None and model and vocab and resnet:
        # Generate caption if not cached
        if st.session_state.caption is None:
            bar = st.progress(0)
            for i in range(100):
                bar.progress(i + 1)
            st.session_state.caption = generate_caption(image, resnet, model, vocab, method='beam')
            bar.empty()

        st.markdown(f"""
        <div class="ns-caption-area">
            <p class="ns-caption-result">{st.session_state.caption}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="ns-caption-area">
            <p class="ns-caption-placeholder">Caption will appear here</p>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ──
st.markdown('<p class="ns-footer">Powered by PyTorch · ResNet-50 · LSTM</p>', unsafe_allow_html=True)