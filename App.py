# App.py — Bloom Question Type Analyzer (Gujarati/Hindi/English) ✅ HCI UI + Donut Hover Questions
# - Clean sidebar (only Language dropdown)
# - Auto-loads model based on language
# - One chart only (Donut). Hover shows questions under that class.

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
    page_title="Bloom Question Type Analyzer",
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
textarea { border-radius: 14px !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🧠 Bloom Question Type Analyzer")
st.markdown(
    "<div class='helper'>Paste questions (line-by-line or paragraph). "
    "Select language and click <b>Analyze</b> to predict type, group results, and visualize distribution.</div>",
    unsafe_allow_html=True
)

# ---------------------------
# 2) Text Utilities (Language-aware)
# ---------------------------
def clean_text(text: str, lang: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    if lang == "Gujarati":
        # Gujarati block + digits + English + punctuation
        text = re.sub(r"[^૦-૯0-9\u0A80-\u0AFFA-Za-z\s\?\.\,\-\/\(\)%:;'\u0964]", " ", text)
    elif lang == "Hindi":
        # Devanagari block + digits + English + punctuation + danda
        text = re.sub(r"[^0-9०-९\u0900-\u097FA-Za-z\s\?\.\,\-\/\(\)%:;'\u0964]", " ", text)
    else:  # English
        # Keep letters/digits/basic punctuation (don't over-clean)
        text = re.sub(r"[^A-Za-z0-9\s\?\.\,\-\/\(\)%:;']", " ", text)

    text = re.sub(r"(\?){2,}", "?", text)
    text = re.sub(r"(\.){2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_questions(raw: str, lang: str):
    """
    Supports:
    - line-by-line input
    - paragraph input
    Splits by newline then by '?'
    """
    if not raw:
        return []
    raw = raw.replace("।", ".")  # normalize danda to dot for splitting consistency
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

    final, seen = [], set()
    for q in qs:
        q2 = clean_text(q, lang)
        if q2 and q2 not in seen:
            final.append(q2)
            seen.add(q2)
    return final

# ---------------------------
# 3) Robust Model Loading + Prediction
# ---------------------------
MODEL_FILES = {
    "Gujarati": "Correct_model_gujarati_bloom_lr_tfidf.joblib",
    "Hindi": "Correct_model_Hindi_bloom_lr_tfidf.joblib",
    "English": "Correct_model_English_bloom_lr_tfidf.joblib",
}

@st.cache_resource
def load_raw(path: str):
    return joblib.load(path)

def normalize_loaded_object(obj):
    if hasattr(obj, "predict"):
        return {"kind": "pipeline", "pipeline": obj}

    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("clf") or obj.get("classifier")
        vect  = obj.get("vectorizer") or obj.get("tfidf") or obj.get("vect")
        if model is not None and vect is not None and hasattr(model, "predict") and hasattr(vect, "transform"):
            return {"kind": "vect_model", "model": model, "vectorizer": vect}
        return {"kind": "unknown_dict", "keys": list(obj.keys()), "raw": obj}

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
            f"Model is a dict with keys: {loaded['keys']} (unsupported). "
            f"Save as a Pipeline OR dict with ('vectorizer','model')."
        )
    raise AttributeError("Unsupported model object. Save as sklearn Pipeline or (vectorizer+model).")

# ---------------------------
# 4) Sidebar (Only Language dropdown — no extra text)
# ---------------------------
lang = st.sidebar.selectbox(
    "Language",
    ["Gujarati", "Hindi", "English"],
    index=0,
    label_visibility="collapsed"
)

model_path = MODEL_FILES[lang]
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}\n\nUpload it in the same folder as App.py.")
    st.stop()

try:
    raw_obj = load_raw(model_path)
    loaded = normalize_loaded_object(raw_obj)
    if loaded["kind"] in ("unknown", "unknown_dict"):
        st.error("❌ Model format not supported. Please save as Pipeline or (vectorizer+model).")
        st.stop()
except Exception as e:
    st.error("❌ Model could not be loaded.")
    st.exception(e)
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

    placeholder_map = {
        "Gujarati": "ઉદાહરણ (એક લાઇનમાં એક પ્રશ્ન લખો):\n"
                    "AI મોડલ ટ્રેનિંગમાં ડેટા બાયસ કેમ થાય છે?\n"
                    "કેશ પેમેન્ટ અને UPI વચ્ચે તફાવત શું છે?\n"
                    "જો ગ્લોબલ વોર્મિંગ 2°C વધે તો કૃષિ પર શું થશે?\n"
                    "તમે ઘર માટે પાણી બચત યોજના 3 પગલાંમાં બનાવો.",
        "Hindi": "उदाहरण (एक लाइन में एक प्रश्न लिखें):\n"
                 "AI मॉडल प्रशिक्षण में डेटा बायस क्यों होता है?\n"
                 "नकद भुगतान और UPI के बीच क्या अंतर है?\n"
                 "यदि ग्लोबल वार्मिंग 2°C बढ़ जाए तो कृषि पर क्या प्रभाव पड़ेगा?\n"
                 "घर के लिए पानी बचत की 3-चरणों वाली योजना बनाइए.",
        "English": "Example (one question per line):\n"
                   "What factors lead to data bias during AI model training?\n"
                   "How do cash payments and UPI differ in practice?\n"
                   "If global warming rises by 2°C, what impacts are likely for agriculture?\n"
                   "Create a 3-step plan for saving water at home."
    }

    raw_input = st.text_area(
        "Questions",
        value="",
        placeholder=placeholder_map[lang],
        height=230,
        label_visibility="collapsed"
    )

    st.markdown(
        "<span class='pill'>Paragraph supported</span>"
        "<span class='pill'>Line-by-line</span>"
        "<span class='pill'>Tech terms supported</span>",
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
        questions = extract_questions(raw_input, lang)
        if not questions:
            st.warning("No valid questions found. Please paste at least one question.")
        else:
            preds, conf = predict_with_confidence(loaded, questions)

            df_out = pd.DataFrame({
                "question": questions,
                "predicted_type": preds
            })

            # Show confidence only if available
            if conf is not None:
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
            with st.expander(f"{cls} (Total: {len(grouped[cls])})", expanded=False):
                for i, q in enumerate(grouped[cls], start=1):
                    st.write(f"{i}. {q}")
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    df_out = st.session_state.df_out
    if df_out is None:
        st.info("No results yet. Analyze some questions to see charts.")
    else:
        # counts + questions per class
        grouped = df_out.groupby("predicted_type")["question"].apply(list).to_dict()
        counts = df_out["predicted_type"].value_counts().sort_values(ascending=False)

        counts_df = pd.DataFrame({
            "type": counts.index,
            "count": counts.values
        })

        # Build hover question list (ALL questions)
        # (HTML line breaks work in Plotly hover)
        hover_q = []
        for t in counts_df["type"]:
            qs = grouped.get(t, [])
            # bullet list in hover
            html = "<br>".join([f"• {q}" for q in qs])
            hover_q.append(html)

        counts_df["questions_html"] = hover_q

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Questions", int(len(df_out)))
        c2.metric("Unique Types", int(df_out["predicted_type"].nunique()))
        c3.metric("Most Frequent Type", str(counts_df.loc[0, "type"]))
        c4.metric("Max Count", int(counts_df.loc[0, "count"]))

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Donut Chart (Hover to see questions)")

        fig = px.pie(
            counts_df,
            names="type",
            values="count",
            hole=0.45
        )

        # Put the questions into hover via customdata
        fig.update_traces(
            customdata=np.stack([counts_df["questions_html"]], axis=-1),
            hovertemplate="<b>%{label}</b><br>"
                          "Count: %{value}<br><br>"
                          "%{customdata[0]}"
                          "<extra></extra>"
        )

        fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Heatmap-like Table")
        heat_df = counts_df.set_index("type")[["count"]].T
        st.dataframe(heat_df.style.background_gradient(axis=None), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='helper'>Interpretation: If one type dominates, your input is biased toward that Bloom level.</div>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("HCI-friendly UI • Gujarati/Hindi/English NLP • Bloom Taxonomy Classification")
