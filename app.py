
import streamlit as st
import joblib, torch, re, numpy as np
from urllib.parse import urlparse
import tldextract
from transformers import AutoTokenizer, AutoModel

st.title("🔍 URL Phishing Detector")

# Load artifacts
tfidf_path = "/content/drive/MyDrive/Colab Notebooks/Paul/tfidf_vectorizer.pkl"
pca_path   = "/content/drive/MyDrive/Colab Notebooks/Paul/pca_roberta.pkl"
model_path = "/content/drive/MyDrive/Colab Notebooks/Paul/best_hybrid_model.pkl"

vectorizer = joblib.load(tfidf_path)
pca        = joblib.load(pca_path)
clf        = joblib.load(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
roberta_model = AutoModel.from_pretrained("distilroberta-base").to(device)
roberta_model.eval()

# Helper functions
SENSITIVE_WORDS = [
    'confirm','account','banking','secure','ebyisapi','webscr','signin',
    'mail','install','toolbar','backup','paypal','password','username'
]

def clean_url(url: str) -> str:
    url = url.lower().strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    url = re.sub(r'[\?#].*', '', url)
    url = url.rstrip('/')
    return url

def get_meta_features(url: str):
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    return np.array([
        int('@' in url),
        int('-' in ext.domain),
        int('//' in (parsed.path or '')),
        int(any(word in url for word in SENSITIVE_WORDS)),
        int(url.count('/') > 5)
    ], dtype=int)

def get_roberta_embedding(url: str):
    with torch.no_grad():
        encoded = tokenizer([url], padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = roberta_model(**encoded)
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return emb

def predict_url(url: str):
    url_clean = clean_url(url)
    meta_feat = get_meta_features(url_clean).reshape(1, -1)
    tfidf_feat = vectorizer.transform([url_clean])
    emb_feat = get_roberta_embedding(url_clean)
    emb_feat_reduced = pca.transform(emb_feat)
    final_feat = np.hstack([tfidf_feat.toarray(), emb_feat_reduced, meta_feat])

    pred_label = clf.predict(final_feat)[0]
    pred_prob  = clf.predict_proba(final_feat)[0, 1]
    label_str = "BAD (Phishing URL)" if pred_label==1 else "GOOD (Legit URL)"
    return label_str, pred_prob

#  Streamlit UI
# Initialize session state for text area
if "url_text" not in st.session_state:
    st.session_state.url_text = ""

def clear_text():
    st.session_state.url_text = ""  # this works as a callback

# Text area
url_input = st.text_area("Enter URL(s) separated by commas:", value=st.session_state.url_text, key="url_text")

# Buttons in two columns
col1, col2, col3 = st.columns([2, 5, 1])

with col1:
    check_btn = st.button("Check URL")
with col3:
    clear_btn = st.button("Clear", on_click=clear_text)

# Check URLs logic
if check_btn and url_input.strip():
    url_list = [u.strip() for u in url_input.split(",") if u.strip()]
    for url in url_list:
        try:
            label, prob = predict_url(url)
            if label.startswith("BAD"):
                st.error(f"⚠️ Warning: {url} is likely a phishing site! Probability: {prob*100:.0f}%")
            else:
                st.success(f"✅ Safe: {url} appears legitimate! Probability: {prob*100:.0f}%")
        except Exception as e:
            st.warning(f"⚠️ {url} → Error: {e}")
