# App.py — HCI-friendly UI (Gujarati Bloom Question Type Analyzer) ✅ FIXED + POLISHED
# - Better UI (background, fewer buttons, nicer placeholder)
# - Robust model loading (supports Pipeline OR (vectorizer+model) dict/tuple)
# - Predict many questions (paragraph or line-by-line), group by class
# - Bar chart + Heatmap
#
# Requirements (Streamlit Cloud):
# streamlit
# scikit-learn
# joblib
# pandas
# matplotlib
# numpy

import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------
# 1) Page Config + Styling
# ---------------------------
st.set_page_config(
    page_title="Gujarati Bloom Question Type Analyzer",
    page_icon="🧠",
    layout="wide"
)

CUSTOM_CSS = """
<style>
/* overall background gradient */
.stApp {
  background: radial-gradient(circle at 20% 10%, rgba(99,102,241,0.15) 0%, rgba(0,0,0,0) 45%),
              radial-gradient(circle at 80% 0%, rgba(16,185,129,0.12) 0%, rgba(0,0,0,0) 40%),
              linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(2, 6, 23, 0.95));
}

/* page padding */
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1200px;}

/* title spacing */
h1 {margin-bottom: 0.25rem;}
h2, h3 {margin-top: 0.2rem;}

/* card look */
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}

/* helper text */
.helper {opacity: 0.90; font-size: 0.95rem; line-height: 1.35rem;}

/* pill badge */
.pill {
  display:inline-block;
  padding: 0.22rem 0.60rem;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.06);
  margin-right: 0.35rem;
  margin-top: 0.35rem;
  font-size: 0.85rem;
}

/* make text area nicer */
textarea {
  border-radius: 14px !important;
}

/* reduce button visual noise */
.stButton>button {
  border-radius: 12px;
  padding: 0.55rem 0.85rem;
  font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Header
st.title("🧠 Gujarati Bloom Question Type Analyzer")
st.markdown(
    "<div class='helper'>Paste Gujarati questions (line-by-line or paragraph). "
    "Click <b>Analyze</b> to predict type, group results, and view distribution (bar + heatmap).</div>",
    unsafe_allow_html=True
)

# ---------------------------
# 2) Text Utilities
# ---------------------------
def clean_gujarati_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # keep Gujarati + digits + basic punctuation + English letters (AI, GDP, UPI etc.)
    text = re.sub(r"[^૦-૯0-9\u0A80-\u0AFFA-Za-z\s\?\.\,\-\/\(\)%:;']", " ", text)
    text = re.sub(r"(\?){2,}", "?", text)
    text = re.sub(r"(\.){2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_questions(raw: str):
    """
    Works with:
    - line-by-line questions
    - paragraph containing many questions
    Splits on newline first; then splits on '?' while keeping '?'
    """
    if not raw:
        return []
    raw = raw.replace("।", ".")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # if user pasted one big paragraph (no newlines)
    if len(lines) == 0 and raw.strip():
        lines = [raw.strip()]

    questions = []
    for ln in lines:
        parts = re.split(r"(\?)", ln)
        buf = ""
        for p in parts:
            if p == "?":
                buf = (buf + p).strip()
                if buf:
                    questions.append(buf)
                buf = ""
            else:
                buf = (buf + " " + p).strip()

        # leftover without '?'
        if buf.strip():
            questions.append(buf.strip())

    # clean + dedupe (keep order)
    final, seen = [], set()
    for q in questions:
        q2 = clean_gujarati_text(q)
        if q2 and q2 not in seen:
            final.append(q2)
            seen.add(q2)
    return final

# ---------------------------
# 3) Robust Model Loader (Pipeline OR (vectorizer+model) dict/tuple)
# ---------------------------
@st.cache_resource
def load_any_model(model_path: str):
    obj = joblib.load(model_path)

    # Case A: sklearn Pipeline/estimator with predict
    if hasattr(obj, "predict"):
        return {"kind": "pipeline_or_estimator", "pipe": obj}

    # Case B: dict {"model":..., "vectorizer":...}
    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("clf") or obj.get("classifier")
        vect = obj.get("vectorizer") or obj.get("tfidf") or obj.get("vect")
        if model is not None and vect is not None and hasattr(model, "predict") and hasattr(vect, "transform"):
            return {"kind": "vect+model_dict", "model": model, "vectorizer": vect}

    # Case C: tuple/list (vectorizer, model) or (model, vectorizer)
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        a, b = obj
        if hasattr(a, "transform") and hasattr(b, "predict"):
            return {"kind": "vect+model_tuple", "vectorizer": a, "model": b}
        if hasattr(b, "transform") and hasattr(a, "predict"):
            return {"kind": "vect+model_tuple", "vectorizer": b, "model": a}

    return {"kind": "unknown", "raw": obj}

def safe_predict_with_confidence(loaded, texts):
    """
    loaded: output from load_any_model()
    Returns preds, conf (max prob) or (preds, None)
    """
    texts = list(texts)

    if loaded["kind"] == "pipeline_or_estimator":
        pipe = loaded["pipe"]
        preds = pipe.predict(texts)
        conf = None
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(texts)
            conf = np.max(proba, axis=1)
        return preds, conf

    if loaded["kind"] in ("vect+model_dict", "vect+model_tuple"):
        vect = loaded["vectorizer"]
        clf = loaded["model"]
        X = vect.transform(texts)
        preds = clf.predict(X)
        conf = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)
            conf = np.max(proba, axis=1)
        return preds, conf

    raise AttributeError(
        "Loaded object is not a valid sklearn pipeline/model. "
        "Please upload a correct .joblib that contains Pipeline OR (vectorizer+model)."
    )

# ---------------------------
# 4) Charts
# ---------------------------
def plot_heatmap(counts: pd.Series, title="Heatmap: Type Distribution"):
    labels = counts.index.tolist()
    values = counts.values.tolist()

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.1), 2.8))
    heat = ax.imshow([values], aspect="auto")

    ax.set_yticks([0])
    ax.set_yticklabels(["Count"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")

    for i, v in enumerate(values):
        ax.text(i, 0, str(v), ha="center", va="center")

    ax.set_title(title)
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    return fig

def plot_bar(counts: pd.Series, title="Question Type Counts"):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(counts.index.tolist(), counts.values.tolist())
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    return fig

# ---------------------------
# 5) Sidebar (clean + minimal)
# ---------------------------
st.sidebar.header("⚙️ Settings")

available_models = sorted([f for f in os.listdir(".") if f.lower().endswith((".joblib", ".pkl"))])

if not available_models:
    st.error("❌ No model file found in this folder.\n\nUpload/commit your `.joblib/.pkl` in the same repo folder as `App.py`.")
    st.stop()

MODEL_PATH = st.sidebar.selectbox(
    "Model file",
    options=available_models,
    index=0
)

show_confidence = st.sidebar.toggle("Show confidence", value=True)
expanded_groups = st.sidebar.toggle("Expand groups by default", value=False)

# Load model
try:
    loaded = load_any_model(MODEL_PATH)
    if loaded["kind"] == "unknown":
        st.sidebar.error("❌ Loaded file is not a sklearn model/pipeline.")
        with st.sidebar.expander("Debug info"):
            st.write("Loaded type:", type(loaded["raw"]))
            st.write("Preview:", str(loaded["raw"])[:250])
        st.stop()
    else:
        st.sidebar.success(f"✅ Model loaded ({loaded['kind']})")
except Exception as e:
    st.sidebar.error("❌ Model not loaded. Check file name/path.")
    st.sidebar.exception(e)
    st.stop()

# ---------------------------
# 6) Main Tabs (HCI friendly)
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📝 Analyze", "📊 Results", "💡 Insights"])

if "df_out" not in st.session_state:
    st.session_state.df_out = None

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Input")

    st.markdown(
        "<div class='helper'>Tip: Enter one question per line. Paragraph also works (we split using <b>?</b>).</div>",
        unsafe_allow_html=True
    )

    raw_input = st.text_area(
        "Questions",
        value="",
        placeholder="ઉદાહરણ:\nAI મોડલ ટ્રેનિંગમાં ડેટા બાયસ કેમ થાય છે?\nકેશ પેમેન્ટ અને UPI વચ્ચે તફાવત શું છે?\nજો ગ્લોબલ વોર્મિંગ 2°C વધે તો કૃષિ પર શું થશે?\nતમે ઘર માટે પાણી બચત યોજના 3 પગલાંમાં બનાવો.",
        height=220
    )

    st.markdown(
        "<span class='pill'>Paragraph supported</span>"
        "<span class='pill'>Line-by-line supported</span>"
        "<span class='pill'>Gujarati + Tech terms</span>",
        unsafe_allow_html=True
    )

    colA, colB = st.columns([1, 1])
    with colA:
        analyze = st.button("🔎 Analyze", type="primary", use_container_width=True)
    with colB:
        clear = st.button("🧹 Clear", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if clear:
        st.session_state.df_out = None
        st.success("Cleared results. Paste new questions and click Analyze.")

    if analyze:
        questions = extract_questions(raw_input)
        if not questions:
            st.warning("No valid questions found. Please paste at least one question.")
        else:
            preds, conf = safe_predict_with_confidence(loaded, questions)

            df_out = pd.DataFrame({
                "question": questions,
                "predicted_type": preds
            })

            if show_confidence and conf is not None:
                df_out["confidence"] = np.round(conf, 4)

            st.session_state.df_out = df_out
            st.success(f"✅ Done! Total questions analyzed: {len(df_out)}")

with tab2:
    df_out = st.session_state.df_out
    if df_out is None:
        st.info("No results yet. Go to **Analyze** tab and click **Analyze**.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Predictions")
        st.dataframe(df_out, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Download CSV
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name="predicted_question_types.csv",
            mime="text/csv"
        )

        # Grouped output
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Grouped by Type")
        grouped = df_out.groupby("predicted_type")["question"].apply(list).to_dict()

        for cls in sorted(grouped.keys()):
            with st.expander(f"{cls}  (Total: {len(grouped[cls])})", expanded=expanded_groups):
                for i, q in enumerate(grouped[cls], start=1):
                    st.write(f"{i}. {q}")
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    df_out = st.session_state.df_out
    if df_out is None:
        st.info("No results yet. Analyze some questions to see insights.")
    else:
        counts = df_out["predicted_type"].value_counts().sort_index()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Questions", int(len(df_out)))
        c2.metric("Unique Types", int(df_out["predicted_type"].nunique()))
        c3.metric("Most Frequent Type", str(counts.idxmax()))
        c4.metric("Max Count", int(counts.max()))

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Distribution")

        left, right = st.columns([1, 1])
        with left:
            st.pyplot(plot_bar(counts))
        with right:
            st.pyplot(plot_heatmap(counts))

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='helper'>Interpretation: If one type dominates, your input set is biased toward that Bloom level.</div>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("Made for HCI-friendly UI • Gujarati NLP • Bloom Taxonomy Classification")
