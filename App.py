# app.py
# Streamlit app: Gujarati Bloom Question Type Classifier (LogReg + TF-IDF)
# - Input: multiple questions (paragraph or line-by-line)
# - Output: predicted type per question, grouped by class, and a heatmap of class distribution

import re
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------
# 1) Page Config
# ---------------------------
st.set_page_config(
    page_title="Gujarati Question Type Analyzer (Bloom)",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Gujarati Question Type Analyzer (Bloom Taxonomy)")
st.caption("Paste multiple Gujarati questions (paragraph or line-by-line). Click **Analyze Type** to classify and visualize distribution.")

# ---------------------------
# 2) Load Model (joblib/pkl)
# ---------------------------
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

MODEL_PATH = st.sidebar.text_input(
    "Model file path (.joblib / .pkl)",
    value="Correct_model_gujarati_bloom_lr_tfidf.joblib"
)

try:
    model = load_model(MODEL_PATH)
    st.sidebar.success("✅ Model loaded")
except Exception as e:
    st.sidebar.error("❌ Model not loaded. Check file path/name.")
    st.sidebar.exception(e)
    st.stop()

# ---------------------------
# 3) Cleaning + Parsing
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
    Accepts paragraph or line-by-line.
    Strategy:
    1) Split by newline as primary units
    2) For each line, further split by '?' keeping the question mark
    3) If no '?', keep the line as a question if it's non-empty
    """
    raw = raw.replace("।", ".")  # optional normalization
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    questions = []
    for ln in lines:
        # Split by '?', but keep it
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

        # If leftover text (no '?') treat as one question/sentence
        leftover = buf.strip()
        if leftover:
            questions.append(leftover)

    # If user pasted one big paragraph without newlines:
    if not questions and raw.strip():
        tmp = raw.strip()
        parts = re.split(r"(\?)", tmp)
        buf = ""
        for p in parts:
            if p == "?":
                buf = (buf + p).strip()
                if buf:
                    questions.append(buf)
                buf = ""
            else:
                buf = (buf + " " + p).strip()
        if buf.strip():
            questions.append(buf.strip())

    # Final clean + dedupe while keeping order
    final = []
    seen = set()
    for q in questions:
        q2 = clean_gujarati_text(q)
        if q2 and q2 not in seen:
            final.append(q2)
            seen.add(q2)
    return final

def plot_heatmap(counts: pd.Series):
    """
    Heatmap for class distribution.
    Makes a 1 x N heatmap (single-row) with annotations.
    """
    labels = counts.index.tolist()
    values = counts.values.tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.2), 2.5))
    heat = ax.imshow([values], aspect="auto")  # 1-row heatmap

    # Ticks
    ax.set_yticks([0])
    ax.set_yticklabels(["Count"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")

    # Annotate values
    for i, v in enumerate(values):
        ax.text(i, 0, str(v), ha="center", va="center")

    ax.set_title("Heatmap: Question Type Distribution")
    fig.colorbar(heat, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    return fig

# ---------------------------
# 4) UI Input
# ---------------------------
default_text = """AI મોડલ ટ્રેનિંગમાં ડેટા બાયસ કેમ થાય છે?
કેશ પેમેન્ટ અને UPI વચ્ચે તફાવત શું છે?
જો ગ્લોબલ વોર્મિંગ 2°C વધે તો કૃષિ પર શું થશે?
તમે ઘર માટે પાણી બચત યોજના 3 પગલાંમાં બનાવો."""
raw_input = st.text_area(
    "✍️ Paste questions (paragraph or line-by-line):",
    value=default_text,
    height=220
)

analyze = st.button("🔎 Analyze Type", type="primary")

# ---------------------------
# 5) Prediction + Grouping + Visuals
# ---------------------------
if analyze:
    questions = extract_questions(raw_input)

    if not questions:
        st.warning("No valid questions found. Please enter at least one question.")
        st.stop()

    # Predict
    preds = model.predict(questions)

    df_out = pd.DataFrame({
        "question": questions,
        "predicted_type": preds
    })

    # Summary counts
    counts = df_out["predicted_type"].value_counts().sort_index()

    # Layout
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("📋 Predictions (per question)")
        st.dataframe(df_out, use_container_width=True, hide_index=True)

        st.subheader("📌 Grouped by Type")
        grouped = df_out.groupby("predicted_type")["question"].apply(list).to_dict()

        # Display each group in expandable sections
        for cls in sorted(grouped.keys()):
            with st.expander(f"{cls}  (Total: {len(grouped[cls])})", expanded=(len(grouped[cls]) > 0)):
                for i, q in enumerate(grouped[cls], start=1):
                    st.write(f"{i}. {q}")

    with right:
        st.subheader("📊 Distribution")
        st.write("Counts by class:")
        st.dataframe(counts.reset_index().rename(columns={"index": "type", "predicted_type": "count"}),
                     use_container_width=True, hide_index=True)

        # Bar chart (extra clarity)
        st.subheader("📈 Bar Chart")
        fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
        ax_bar.bar(counts.index.tolist(), counts.values.tolist())
        ax_bar.set_ylabel("Count")
        ax_bar.set_title("Question Type Counts")
        ax_bar.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig_bar)

        # Heatmap
        st.subheader("🧩 Heatmap")
        fig_hm = plot_heatmap(counts)
        st.pyplot(fig_hm)

    st.success(f"✅ Done! Total questions analyzed: {len(df_out)}")

# ---------------------------
# 6) Footer Tips
# ---------------------------
with st.expander("ℹ️ Tips"):
    st.write(
        "- Put each question on a new line for best results.\n"
        "- You can also paste a paragraph; the app will split using '?' and newlines.\n"
        "- Keep your model file in the same folder as app.py OR give the correct path in the sidebar."
    )
