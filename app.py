"""
Stress & Depression Detector - Deep Learning-Based Detection of Stress and Depression from Text
Streamlit front end for a Bi-LSTM text classifier (Final Year Project).

Run:  streamlit run app.py
Needs Streamlit >= 1.46 (keyed containers + st.context.theme).

Optional: create .streamlit/config.toml with
    [client]
    toolbarMode = "developer"
to show the toolbar / menu (setting it at runtime from code does not work).
"""

import logging
import pickle
from pathlib import Path

import streamlit as st
import tensorflow as tf

try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:  # newer Keras 3 builds
    from keras.utils import pad_sequences

from preprocessing import clean_text

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "stress_model.h5"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"
MAX_LEN = 150          # must match the value used during training
THRESHOLD = 0.5
MAX_CHARS = 1500

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("psytext")

EXAMPLES = {
    "Overwhelmed": "I can't keep up anymore. Every day feels heavier than the last and I'm exhausted all the time.",
    "Hopeful": "Had a great weekend hiking with friends. Feeling rested and looking forward to the week ahead!",
    "Anxious": "My heart races every time I think about tomorrow's deadline. I can't sleep and I can't focus.",
}

# set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="Stress & Depression Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --charcoal: #232623;
    --charcoal-2: #2F332F;
    --charcoal-3: #454A45;
    --lime: #C5F135;
    --lime-dark: #A9D41C;
    --grey-bg: #ECEEEB;
    --grey-card: #F7F8F6;
    --grey-line: #D9DCD7;
    --grey-text: #6B716B;
    --coral: #FF7F5C;
    --ink: #232623;
    --field: #FFFFFF;
    --empty-line: #C3C7C1;
    --empty-bg: rgba(247,248,246,.6);
}

html, body, [class*="css"], .stApp, textarea, button {
    font-family: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}

.stApp { background: var(--grey-bg); color: var(--ink); }
header[data-testid="stHeader"] { background: var(--grey-bg); z-index: 999990; }
[data-testid="stToolbar"], [data-testid="stMainMenu"],
[data-testid="stAppDeployButton"], .stDeployButton {
    visibility: visible !important;
    opacity: 1 !important;
}
footer { display: none !important; }
.block-container { max-width: 1180px; padding: 4.2rem 1.5rem 3rem; }

.brand { display: flex; align-items: center; gap: .7rem; padding: 0 .2rem; }
.brand-mark {
    width: 42px; height: 42px; border-radius: 12px;
    background: var(--charcoal);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem;
    box-shadow: inset 0 -4px 0 var(--lime);
}
.brand-name { font-size: 1.3rem; font-weight: 800; color: var(--ink); letter-spacing: -.02em; line-height: 1; }
.brand-name span { background: var(--lime); color: #232623; padding: 0 .3rem; border-radius: 6px; }
.brand-sub { font-size: .78rem; font-weight: 600; color: var(--grey-text); margin-top: .3rem; line-height: 1.3; }

.hero {
    background: var(--charcoal);
    border-radius: 24px;
    padding: 2.6rem 2.8rem;
    margin: 1.2rem 0 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    right: -70px; top: -70px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: var(--lime);
    opacity: .95;
}
.hero-badge {
    display: inline-block;
    background: rgba(197,241,53,.14);
    color: var(--lime);
    font-weight: 700;
    font-size: .8rem;
    padding: .35rem .8rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}
.hero h1 {
    color: #fff;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -.02em;
    line-height: 1.12;
    margin: 0 0 .8rem;
    max-width: 640px;
    padding: 0;
}
.hero p {
    color: #B9BEB8;
    font-size: 1.02rem;
    line-height: 1.6;
    max-width: 560px;
    margin: 0;
}

.panel-title { font-size: 1.15rem; font-weight: 800; color: var(--ink); margin: 0 0 .25rem; }
.panel-sub   { font-size: .9rem; color: var(--grey-text); margin: 0 0 1rem; }

.st-key-input_card {
    background: var(--grey-card);
    border: 1px solid var(--grey-line);
    border-radius: 20px;
    padding: 1.6rem 1.6rem 1.4rem;
}
.st-key-input_card textarea {
    background: var(--field) !important;
    color: var(--ink) !important;
    border: 1.5px solid var(--grey-line) !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    line-height: 1.55 !important;
    padding: 1rem !important;
    min-height: 220px;
}
.st-key-input_card textarea:focus {
    border-color: var(--ink) !important;
    box-shadow: 0 0 0 4px rgba(197,241,53,.55) !important;
}
.try-label { font-size: .85rem; font-weight: 700; color: var(--grey-text); margin: .9rem 0 .4rem; }

.st-key-input_card button[kind="primary"],
.st-key-input_card button[data-testid="stBaseButton-primary"] {
    background: var(--lime);
    color: var(--charcoal);
    border: none;
    border-radius: 12px;
    font-weight: 800;
    font-size: 1rem;
    padding: .7rem 1.2rem;
    width: 100%;
}
.st-key-input_card button[kind="primary"]:hover,
.st-key-input_card button[data-testid="stBaseButton-primary"]:hover {
    background: var(--lime-dark);
}
/* Streamlit puts the label inside <p>, so colour it explicitly (contrast fix) */
.st-key-input_card button[kind="primary"] p,
.st-key-input_card button[data-testid="stBaseButton-primary"] p {
    color: var(--charcoal) !important;
}
.st-key-input_card button[kind="secondary"],
.st-key-input_card button[data-testid="stBaseButton-secondary"] {
    background: var(--field);
    color: var(--ink);
    border: 1.5px solid var(--grey-line);
    border-radius: 999px;
    font-weight: 600;
    font-size: .85rem;
    padding: .25rem .9rem;
    width: 100%;
}
.st-key-input_card button[kind="secondary"] p,
.st-key-input_card button[data-testid="stBaseButton-secondary"] p {
    color: var(--ink) !important;
}

.result {
    background: var(--charcoal);
    border-radius: 20px;
    padding: 1.8rem;
    color: #fff;
}
.result.calm   { --accent: var(--lime); }
.result.stress { --accent: var(--coral); }
.result-chip {
    display: inline-flex; align-items: center; gap: .45rem;
    background: rgba(255,255,255,.08);
    color: var(--accent);
    font-weight: 700; font-size: .85rem;
    padding: .35rem .85rem; border-radius: 999px;
}
.result-chip i { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); display: inline-block; }
.result h2 { color: #fff; font-size: 1.7rem; font-weight: 800; letter-spacing: -.01em; margin: 1rem 0 .3rem; padding: 0; }
.result .lead { color: #B9BEB8; font-size: .95rem; line-height: 1.55; margin: 0 0 1.4rem; }

.score-row { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: .6rem; }
.score-big { font-size: 3.2rem; font-weight: 800; color: var(--accent); line-height: 1; letter-spacing: -.03em; }
.score-label { font-size: .85rem; color: #B9BEB8; font-weight: 600; text-align: right; }

.meter { position: relative; height: 12px; background: var(--charcoal-3); border-radius: 999px; }
.meter-fill { height: 100%; border-radius: 999px; background: var(--accent); }
.meter-tick { position: absolute; left: 50%; top: -5px; width: 2px; height: 22px; background: #fff; opacity: .55; }
.meter-scale { display: flex; justify-content: space-between; font-size: .75rem; color: #8E948E; margin-top: .5rem; }

.stats { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; margin-top: 1.5rem; }
.stat { background: var(--charcoal-2); border-radius: 14px; padding: .9rem 1rem; }
.stat b { display: block; font-size: 1.15rem; color: #fff; font-weight: 800; }
.stat span { font-size: .8rem; color: #9DA39D; }

.notice { margin-top: 1.2rem; font-size: .8rem; line-height: 1.5; color: #9DA39D; border-top: 1px solid var(--charcoal-3); padding-top: 1rem; }

.empty {
    border: 2px dashed var(--empty-line);
    border-radius: 20px;
    padding: 3rem 1.8rem;
    text-align: center;
    color: var(--grey-text);
    background: var(--empty-bg);
}
.empty .icon { font-size: 2.4rem; margin-bottom: .6rem; }
.empty h3 { color: var(--ink); font-size: 1.1rem; font-weight: 800; margin: 0 0 .3rem; padding: 0; }
.empty p { font-size: .92rem; margin: 0; line-height: 1.55; }

.footer { text-align: center; color: var(--grey-text); font-size: .8rem; margin-top: 2rem; line-height: 1.6; }

@media (max-width: 640px) {
    .hero { padding: 1.8rem 1.4rem; }
    .hero h1 { font-size: 1.8rem; }
    .score-big { font-size: 2.6rem; }
}
</style>
"""

DARK_CSS = """
<style>
:root {
    --ink: #ECEEEB;
    --charcoal: #2A2E2A;
    --charcoal-2: #353A35;
    --charcoal-3: #4A504A;
    --grey-bg: #131513;
    --grey-card: #1E211E;
    --grey-line: #383D38;
    --grey-text: #9DA39D;
    --field: #141614;
    --empty-line: #3F453F;
    --empty-bg: rgba(30,33,30,.6);
}
.hero, .result { border: 1px solid #3A3F3A; }
</style>
"""


def current_theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


st.markdown(CSS, unsafe_allow_html=True)
if current_theme() == "dark":
    st.markdown(DARK_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_trained_model():
    if not MODEL_PATH.exists():
        logger.error("Model file missing: %s", MODEL_PATH)
        return None
    try:
        # compile=False: we only run inference, avoids optimizer/metric warnings
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception:
        logger.exception("Error loading model")
        return None


@st.cache_resource(show_spinner="Loading tokenizer...")
def load_tokenizer():
    if not TOKENIZER_PATH.exists():
        logger.error("Tokenizer file missing: %s", TOKENIZER_PATH)
        return None
    try:
        with open(TOKENIZER_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        logger.exception("Error loading tokenizer")
        return None


model = load_trained_model()
saved_tokenizer = load_tokenizer()

if model is None:
    st.error(f"Model file not found or could not be loaded. Check `{MODEL_PATH}`.")
if saved_tokenizer is None:
    st.error(
        f"Tokenizer file not found or could not be loaded. Check `{TOKENIZER_PATH}`. "
        "Predictions need the same tokenizer that was used in training."
    )


# ─────────────────────────────────────────────────────────────
# Logic
# ─────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    if model is None:
        return {"error": "model_not_loaded"}
    if saved_tokenizer is None:
        # A tokenizer fitted on one sentence gives meaningless predictions,
        # so we refuse instead of silently returning wrong results.
        return {"error": "tokenizer_not_loaded"}

    cleaned = clean_text(text)
    if not cleaned or not cleaned.strip():
        return {"error": "empty_after_clean"}

    seq = saved_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    try:
        prob = float(model.predict(padded, verbose=0)[0][0])
    except Exception:
        logger.exception("Prediction error")
        return {"error": "prediction_failed"}

    return {
        "prob": prob,
        "is_stress": prob > THRESHOLD,
        "words": len(text.split()),
    }


def confidence_label(prob: float) -> str:
    c = max(prob, 1 - prob)
    if c >= 0.85:
        return "High"
    if c >= 0.65:
        return "Moderate"
    return "Low"


def set_example(key: str):
    st.session_state.user_text = EXAMPLES[key]
    st.session_state.result = None


def clear_all():
    st.session_state.user_text = ""
    st.session_state.result = None


def reset_result():
    """Called when the user edits the text, so an old result is not shown for new text."""
    st.session_state.result = None


def render_result(r: dict):
    prob = r["prob"]
    pct = round(prob * 100, 1)
    if r["is_stress"]:
        cls, chip = "stress", "Indicators detected"
        title = "Stress is likely present"
        lead = "This text contains language often linked with stress, anxiety or low mood."
    else:
        cls, chip = "calm", "No indicators detected"
        title = "This text reads as calm"
        lead = "This text doesn't show the language patterns usually linked with stress."

    html = (
        f'<div class="result {cls}">'
        f'<span class="result-chip"><i></i>{chip}</span>'
        f'<h2>{title}</h2>'
        f'<p class="lead">{lead}</p>'
        f'<div class="score-row">'
        f'<div class="score-big">{pct:.0f}%</div>'
        f'<div class="score-label">stress level</div>'
        f'</div>'
        f'<div class="meter"><div class="meter-fill" style="width:{pct}%"></div>'
        f'<div class="meter-tick"></div></div>'
        f'<div class="meter-scale"><span>Low</span><span>High</span></div>'
        f'<div class="stats">'
        f'<div class="stat"><b>{confidence_label(prob)}</b><span>Confidence</span></div>'
        f'<div class="stat"><b>{r["words"]}</b><span>Words checked</span></div>'
        f'</div>'
        f'<p class="notice">This detector is an automated check, not a medical diagnosis. '
        f'If you or someone you know is struggling, please talk to a mental health '
        f'professional or a local crisis service.</p>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Page UI
# ─────────────────────────────────────────────────────────────
st.session_state.setdefault("user_text", "")
st.session_state.setdefault("result", None)

st.markdown(
    '<div class="brand">'
    '<div class="brand-mark">🧠</div>'
    '<div>'
    '<div class="brand-name">Stress &amp; Depression <span>Detector</span></div>'
    '<div class="brand-sub">Deep Learning-Based Detection of Stress and Depression from Text</div>'
    '</div>'
    '</div>'
    '<div class="hero">'
    '<span class="hero-badge">Instant results</span>'
    '<h1>Is stress showing up in your words?</h1>'
    '<p>Paste a post, comment or message. The detector reads the language and shows '
    'how strongly it points to stress.</p>'
    '</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 1], gap="large")

with left:
    with st.container(key="input_card"):
        st.markdown(
            '<p class="panel-title">Your text</p>'
            '<p class="panel-sub">Write or paste a post. Longer text gives a more stable result.</p>',
            unsafe_allow_html=True,
        )
        st.text_area(
            "Text to analyse",
            key="user_text",
            height=230,
            max_chars=MAX_CHARS,
            placeholder="Type or paste text here…",
            label_visibility="collapsed",
            on_change=reset_result,
        )

        st.markdown('<p class="try-label">Or try an example</p>', unsafe_allow_html=True)
        ex_cols = st.columns(len(EXAMPLES) + 1)
        for col, name in zip(ex_cols, EXAMPLES):
            col.button(name, key=f"ex_{name}", on_click=set_example, args=(name,))
        ex_cols[-1].button("Clear", key="clear", on_click=clear_all)

        st.write("")
        analyze = st.button("Analyze text", type="primary", key="analyze")

    if analyze:
        text = st.session_state.user_text
        if not text.strip():
            st.session_state.result = None
            st.warning("Enter some text first, then select Analyze text.")
        else:
            with st.spinner("Analysing…"):
                st.session_state.result = predict(text)

with right:
    result = st.session_state.result
    if result is None:
        st.markdown(
            '<div class="empty">'
            '<div class="icon">📝</div>'
            '<h3>Your result will appear here</h3>'
            '<p>Add some text on the left and select <b>Analyze text</b> to see '
            'your stress level.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif result.get("error") == "empty_after_clean":
        st.warning("There was no usable text left after cleaning. Try a longer or different post.")
    elif result.get("error") == "model_not_loaded":
        st.error("Model could not be loaded. Check if the model file path is correct.")
    elif result.get("error") == "tokenizer_not_loaded":
        st.error("Tokenizer could not be loaded. Check if the tokenizer file path is correct.")
    elif result.get("error") == "prediction_failed":
        st.error("An error occurred during prediction.")
    else:
        render_result(result)

st.markdown(
    '<p class="footer">Stress &amp; Depression Detector · Final Year Project<br>'
    'Results are automated estimates, not a medical diagnosis, and do not replace '
    'professional advice.</p>',
    unsafe_allow_html=True,
)