# App.py — Gujarati Bloom Question Type Analyzer (HCI UI + Sunburst) ✅
# Works with saved model as:
# 1) Pipeline/Estimator (has .predict)
# 2) dict {"vectorizer":..., "model":...} (or similar keys)
# 3) tuple/list (vectorizer, model)

import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------
# 1) Page Config + Styling (Light + Attractive)
# ---------------------------
st.set_page_config(
    page_title="Gujarati Bloom Question Type Analyzer",
    page_icon="🧠",
    layout="wide"
)

CUSTOM_CSS = """
<style>
/* Light, cool background */
.stApp {
  background:
    radial-gradient(circle at 15% 15%, rgba(99,102,241,0.18) 0%, rgba(255,255,255,0) 45%),
    radial-gradient(circle at 85% 10%, rgba(16,185,129,0.14) 0%, rgba(255,255,255,0) 40%),
    radial-gradient(circle at 70% 90%, rgba(236,72,153,0.12) 0%, rgba(255,255,255,0) 45%),
    linear-gradient(180deg, #F8FAFF 0%, #F4FBFF 45%, #F9F5FF 100%);
}

/* Main container */
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1200px;}

/* Titles */
h1 {margin-bottom: 0.25rem; color: #0F172A;}
h2, h3 {color: #0F172A;}
p, li, label, div {color: #0F172A;}

/* Card */
.card {
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(15,23,42,0.10);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 14px 32px rgba(2,6,23,0.08);
  backdrop-filter: blur(6px);
}

/* Helper text */
.helper {opacity: 0.90; font-size: 0.95rem; line-height: 1.35rem; color: #334155;}

/* Pill */
.pill {
  display:inline-block;
  padding: 0.22rem 0.60rem;
  border-radius: 999px;
  border: 1px solid rgba(15,23,42,0.12);
  background: rgba(255,255,255,0.75);
  margin-right: 0.35rem;
  margin-top: 0.35rem;
  font-size: 0.85rem;
  color: #0F172A;
}

/* Buttons */
.stButton>button {
  border-radius: 12px;
  padding: 0.55rem 0.85rem;
  font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.70) !important;
  border-right: 1px solid rgba(15,23,42,0.08);
}

/* Textarea */
textarea {
  border-radius: 14px !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Header
st.title("🧠 Gujarati Bloom Question Type Analyzer")
st.markdown(
    "<div class='helper'>Paste Gujarati questions (line-by-line or paragraph). "
    "Click <b>Analyze</b> to predict type, group results, and visualize distribution.</div>",
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
    # keep Gujarati + digits + punctuation + English letters (AI/GDP/UPI)
    text = re.sub(r"[^૦-૯0-9\u0A80-\u0AFFA-Za-z\s\?\.\,\-\/\(\)%:;']", " ", text)
    text = re.sub(r"(\?){2,}", "?", text)
    text = re.sub(r"(\.){2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_questions(raw: str):
    """
    Supports:
    - line-by-line input
    - paragraph input
    Splits by newline then by '?' while keeping '?'
    """
    if not raw:
        return []
    raw = raw.replace("।", ".")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines and raw.strip():
        lines = [raw.strip()]

    qs = []
    for ln in lines:
        parts = re.split(r"(\?)", ln)
        buf = ""
        for p in parts:
            if p == "?":
                buf = (buf + p).strip()
                if buf:
                    qs.append(buf)
                buf = ""
            else:
                buf = (buf + " " + p).strip()
        if buf.strip():
            qs.append(buf.strip())

    # clean + dedupe
    final, seen = [], set()
    for q in qs:
        q2 = clean_gujarati_text(q)
        if q2 and q2 not in seen:
            final.append(q2)
            seen.add(q2)
    return final

# ---------------------------
# 3) Robust Model Loading + Prediction
# ---------------------------
@st.cache_resource
def load_raw(model_path: str):
    return joblib.load(model_path)

def normalize_loaded_object(obj):
    # A) direct pipeline/estimator
    if hasattr(obj, "predict"):
        return {"kind": "pipeline", "pipeline": obj}

    # B) dict with model + vectorizer
    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("clf") or obj.get("classifier")
        vect  = obj.get("vectorizer") or obj.get("tfidf") or obj.get("vect")
        if model is not None and vect is not None and hasattr(model, "predict") and hasattr(vect, "transform"):
            return {"kind": "vect_model", "model": model, "vectorizer": vect}
        return {"kind": "unknown_dict", "raw": obj, "keys": list(obj.keys())}

    # C) tuple/list (vectorizer, model) or (model, vectorizer)
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        a, b = obj
        if hasattr(a, "transform") and hasattr(b, "predict"):
            return {"kind": "vect_model", "vectorizer": a, "model": b}
        if hasattr(b, "transform") and hasattr(a, "predict"):
            return {"kind": "vect_model", "vectorizer": b, "model": a}

    return {"kind": "unknown", "raw": obj}

def predict_with_confidence(loaded, texts):
    texts = list(texts)

    if loaded["kind"] == "pipeline":
        pipe = loaded["pipeline"]
        preds = pipe.predict(texts)
        conf = None
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(texts)
            conf = np.max(proba, axis=1)
        return preds, conf

    if loaded["kind"] == "vect_model":
        vect = loaded["vectorizer"]
        clf = loaded["model"]
        X = vect.transform(texts)
        preds = clf.predict(X)
        conf = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)
            conf = np.max(proba, axis=1)
        return preds, conf

    if loaded["kind"] == "unknown_dict":
        raise AttributeError(
            f"Your .joblib contains a dict with keys: {loaded['keys']}. "
            f"Expected keys like ('vectorizer','model') or a Pipeline with .predict()."
        )

    raise AttributeError(
        "Loaded object is not a valid sklearn Pipeline OR (vectorizer+model). "
        "Please re-save your model properly."
    )

# ---------------------------
# 4) Sidebar
# ---------------------------
st.sidebar.header("⚙️ Settings")

available_models = sorted([f for f in os.listdir(".") if f.lower().endswith((".joblib", ".pkl"))])
if not available_models:
    st.error("❌ No .joblib/.pkl found in this folder. Upload/commit your model file into the repo.")
    st.stop()

MODEL_PATH = st.sidebar.selectbox("Model file", available_models, index=0)
show_confidence = st.sidebar.toggle("Show confidence", value=True)
expand_groups = st.sidebar.toggle("Expand groups", value=False)

# Load and normalize model
try:
    raw_obj = load_raw(MODEL_PATH)
    loaded = normalize_loaded_object(raw_obj)
    if loaded["kind"] in ("unknown", "unknown_dict"):
        st.sidebar.error("❌ Model format not supported.")
        with st.sidebar.expander("Debug info"):
            st.write("Loaded kind:", loaded["kind"])
            if loaded["kind"] == "unknown_dict":
                st.write("Keys:", loaded.get("keys"))
            st.write("Type:", type(raw_obj))
        st.stop()
    st.sidebar.success(f"✅ Model loaded ({loaded['kind']})")
except Exception as e:
    st.sidebar.error("❌ Model could not be loaded.")
    st.sidebar.exception(e)
    st.stop()

# ---------------------------
# 5) Main Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📝 Analyze", "📊 Results", "✨ Visual Insights"])

if "df_out" not in st.session_state:
    st.session_state.df_out = None

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Input Questions")

    raw_input = st.text_area(
        "Questions",
        value="",
        placeholder="ઉદાહરણ (એક લાઇનમાં એક પ્રશ્ન લખો):\n"
                    "AI મોડલ ટ્રેનિંગમાં ડેટા બાયસ કેમ થાય છે?\n"
                    "કેશ પેમેન્ટ અને UPI વચ્ચે તફાવત શું છે?\n"
                    "જો ગ્લોબલ વોર્મિંગ 2°C વધે તો કૃષિ પર શું થશે?\n"
                    "તમે ઘર માટે પાણી બચત યોજના 3 પગલાંમાં બનાવો.",
        height=230
    )

    st.markdown(
        "<span class='pill'>Paragraph supported</span>"
        "<span class='pill'>Line-by-line</span>"
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
            preds, conf = predict_with_confidence(loaded, questions)

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

        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name="predicted_question_types.csv",
            mime="text/csv"
        )

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Grouped by Type")
        grouped = df_out.groupby("predicted_type")["question"].apply(list).to_dict()

        for cls in sorted(grouped.keys()):
            with st.expander(f"{cls} (Total: {len(grouped[cls])})", expanded=expand_groups):
                for i, q in enumerate(grouped[cls], start=1):
                    st.write(f"{i}. {q}")
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    df_out = st.session_state.df_out
    if df_out is None:
        st.info("No results yet. Analyze some questions to see charts.")
    else:
        counts = df_out["predicted_type"].value_counts().sort_values(ascending=False).reset_index()
        counts.columns = ["type", "count"]

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Questions", int(len(df_out)))
        c2.metric("Unique Types", int(df_out["predicted_type"].nunique()))
        c3.metric("Most Frequent Type", str(counts.loc[0, "type"]))
        c4.metric("Max Count", int(counts.loc[0, "count"]))

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Sunburst (Attractive Bloom Distribution)")

        # Sunburst chart: Root -> Type
        sun_df = pd.DataFrame({
            "Root": ["Bloom Types"] * len(counts),
            "Type": counts["type"],
            "Count": counts["count"]
        })

        fig_sun = px.sunburst(
            sun_df,
            path=["Root", "Type"],
            values="Count",
            title="Bloom Types Distribution (Sunburst)"
        )
        fig_sun.update_layout(margin=dict(t=45, l=10, r=10, b=10))
        st.plotly_chart(fig_sun, use_container_width=True)

        st.subheader("Donut Chart (Quick View)")
        fig_pie = px.pie(
            counts,
            names="type",
            values="count",
            hole=0.45,
            title="Type Share (Donut)"
        )
        fig_pie.update_layout(margin=dict(t=45, l=10, r=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Heatmap-like Table")
        heat_df = counts.set_index("type").T  # one row: count
        styled = heat_df.style.background_gradient(axis=None)
        st.dataframe(styled, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='helper'>Interpretation: If one type dominates, your input is biased toward that Bloom level.</div>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("HCI-friendly UI • Gujarati NLP • Bloom Taxonomy Classification • Sunburst Visualization")
