# App.py — HCI-friendly UI (Gujarati Bloom Question Type Analyzer)
import os
import re
import joblib
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
/* Page padding */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
/* Title styling */
h1 {margin-bottom: 0.25rem;}
/* “Card” look */
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px 16px;
}
/* Small helper text */
.helper {opacity: 0.85; font-size: 0.92rem;}
/* Pill badge */
.pill {
  display:inline-block;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.06);
  margin-right: 0.35rem;
  font-size: 0.85rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🧠 Gujarati Bloom Question Type Analyzer")
st.markdown(
    "<div class='helper'>Paste Gujarati questions (line-by-line or paragraph). "
    "Click <b>Analyze</b> to predict question type, group results, and visualize distribution.</div>",
    unsafe_allow_html=True
)

# ---------------------------
# 2) Utilities
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

    # If user pasted one big paragraph (no newlines), treat as one line
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

        if buf.strip():  # leftover without '?'
            questions.append(buf.strip())

    # Clean + dedupe
    final, seen = [], set()
    for q in questions:
        q2 = clean_gujarati_text(q)
        if q2 and q2 not in seen:
            final.append(q2)
            seen.add(q2)
    return final

def safe_predict_with_confidence(model, texts):
    """
    Returns:
    - preds: predicted labels
    - conf: confidence scores (max prob) if predict_proba exists, else None
    """
    preds = model.predict(texts)
    conf = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)
        conf = probs.max(axis=1)
    return preds, conf

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
# 3) Sidebar: Model + Settings
# ---------------------------
st.sidebar.header("⚙️ Settings")

available_models = [f for f in os.listdir(".") if f.lower().endswith((".joblib", ".pkl"))]
default_model = available_models[0] if available_models else ""

MODEL_PATH = st.sidebar.selectbox(
    "Select model file (.joblib / .pkl)",
    options=available_models if available_models else ["(No model found in folder)"],
    index=0 if available_models else 0
)

st.sidebar.markdown("<div class='helper'>Tip: Put your model file in the same folder as App.py.</div>", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

if not available_models:
    st.error("❌ No model file found in this folder. Upload/commit your .joblib/.pkl into the repo.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    st.sidebar.success("✅ Model loaded")
except Exception as e:
    st.sidebar.error("❌ Model not loaded. Check file name/path.")
    st.sidebar.exception(e)
    st.stop()

show_confidence = st.sidebar.toggle("Show confidence score", value=True)
group_expand_default = st.sidebar.selectbox("Default expand groups", ["Collapsed", "Expanded"], index=0)

# ---------------------------
# 4) Main UI (Tabs)
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📝 Analyze", "📊 Results", "💡 Insights"])

if "df_out" not in st.session_state:
    st.session_state.df_out = None

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Input Questions")

    default_text = """AI મોડલ ટ્રેનિંગમાં ડેટા બાયસ કેમ થાય છે?
કેશ પેમેન્ટ અને UPI વચ્ચે તફાવત શું છે?
જો ગ્લોબલ વોર્મિંગ 2°C વધે તો કૃષિ પર શું થશે?
તમે ઘર માટે પાણી બચત યોજના 3 પગલાંમાં બનાવો."""
    raw_input = st.text_area(
        "Paste questions (paragraph or line-by-line):",
        value=default_text,
        height=200
    )

    colA, colB, colC = st.columns([1, 1, 1.2])
    with colA:
        analyze = st.button("🔎 Analyze", type="primary", use_container_width=True)
    with colB:
        clear = st.button("🧹 Clear results", use_container_width=True)
    with colC:
        st.markdown(
            "<span class='pill'>Supports paragraph</span>"
            "<span class='pill'>Line-by-line</span>"
            "<span class='pill'>Gujarati + Tech terms</span>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if clear:
        st.session_state.df_out = None
        st.success("Cleared results. Now analyze new questions.")

    if analyze:
        questions = extract_questions(raw_input)
        if not questions:
            st.warning("No valid questions found. Please enter at least one question.")
        else:
            preds, conf = safe_predict_with_confidence(model, questions)

            df_out = pd.DataFrame({
                "question": questions,
                "predicted_type": preds
            })

            if show_confidence and conf is not None:
                df_out["confidence"] = conf.round(4)

            st.session_state.df_out = df_out
            st.success(f"✅ Done! Total questions analyzed: {len(df_out)}")

with tab2:
    df_out = st.session_state.df_out
    if df_out is None:
        st.info("No results yet. Go to **Analyze** tab and click **Analyze**.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Predictions (Table)")
        st.dataframe(df_out, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Download
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download results as CSV",
            data=csv,
            file_name="predicted_question_types.csv",
            mime="text/csv"
        )

        # Grouping
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Grouped by Type")
        grouped = df_out.groupby("predicted_type")["question"].apply(list).to_dict()

        expanded_default = (group_expand_default == "Expanded")

        for cls in sorted(grouped.keys()):
            with st.expander(f"{cls}  (Total: {len(grouped[cls])})", expanded=expanded_default):
                for i, q in enumerate(grouped[cls], start=1):
                    st.write(f"{i}. {q}")
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    df_out = st.session_state.df_out
    if df_out is None:
        st.info("No results yet. Analyze some questions to see insights.")
    else:
        counts = df_out["predicted_type"].value_counts().sort_index()

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Questions", int(len(df_out)))
        c2.metric("Unique Types", int(df_out["predicted_type"].nunique()))
        top_type = counts.idxmax()
        c3.metric("Most Frequent Type", top_type)
        c4.metric("Max Count", int(counts.max()))

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Distribution")
        left, right = st.columns([1, 1])

        with left:
            st.pyplot(plot_bar(counts))
        with right:
            st.pyplot(plot_heatmap(counts))

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='helper'>Interpretation tip: If one type dominates, your input set is biased toward that Bloom level.</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Made for HCI-friendly experience • Gujarati NLP • Bloom Taxonomy Classification")
