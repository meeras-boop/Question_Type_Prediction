# Gujarati_Question_Type_Prediction.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import time
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Gujarati Question Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(120deg, #1E3A8A, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        padding: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .confidence-high {
        color: #10B981;
        font-weight: bold;
        font-size: 1.2rem;
        background-color: #ECFDF5;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .confidence-medium {
        color: #F59E0B;
        font-weight: bold;
        font-size: 1.2rem;
        background-color: #FEF3C7;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .confidence-low {
        color: #EF4444;
        font-weight: bold;
        font-size: 1.2rem;
        background-color: #FEE2E2;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .footer {
        text-align: center;
        color: #6B7280;
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.9rem;
        border-top: 1px solid #E5E7EB;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


class GujaratiQuestionClassifier:
    """Simplified classifier that works with both model versions"""
    
    def __init__(self, model_data):
        self.model_data = model_data
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.question_types = model_data['question_types']
        
        # Handle different model versions
        if 'tfidf_vectorizer' in model_data:
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.has_count_vectorizer = 'count_vectorizer' in model_data
            if self.has_count_vectorizer:
                self.count_vectorizer = model_data['count_vectorizer']
        else:
            self.tfidf_vectorizer = model_data['vectorizer']
            self.has_count_vectorizer = False
    
    def predict(self, question):
        """Predict question type"""
        try:
            # Transform question
            features = self.tfidf_vectorizer.transform([question])
            
            # Predict
            pred_encoded = self.classifier.predict(features)[0]
            pred_type = self.label_encoder.inverse_transform([pred_encoded])[0]
            
            # Get probabilities
            probs = self.classifier.predict_proba(features)[0]
            confidence = float(max(probs))
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probs)[-3:][::-1]
            top_3 = [(self.label_encoder.inverse_transform([i])[0], float(probs[i])) 
                     for i in top_3_indices]
            
            # Get all probabilities as dictionary
            prob_dict = {}
            for i, q_type in enumerate(self.question_types):
                prob_dict[q_type] = float(probs[i])
            
            return pred_type, confidence, top_3, prob_dict
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, 0.0, [], {}


@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    model_paths = [
        "enhanced_gujarati_question_classifier.pkl",
        "gujarati_question_classifier.pkl"
    ]
    
    for model_path in model_paths:
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return GujaratiQuestionClassifier(model_data), model_path
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"Error loading {model_path}: {str(e)}")
            continue
    
    return None, None


def get_confidence_info(confidence):
    """Get confidence level styling"""
    if confidence >= 0.8:
        return "confidence-high", "🟢 High Confidence", "#10B981"
    elif confidence >= 0.6:
        return "confidence-medium", "🟡 Medium Confidence", "#F59E0B"
    else:
        return "confidence-low", "🔴 Low Confidence", "#EF4444"


def create_probability_chart(prob_dict):
    """Create probability chart"""
    df = pd.DataFrame({
        'Type': list(prob_dict.keys()),
        'Probability': list(prob_dict.values())
    }).sort_values('Probability', ascending=True)
    
    fig = px.bar(
        df.tail(10),
        x='Probability',
        y='Type',
        orientation='h',
        title="Class Probabilities",
        color='Probability',
        color_continuous_scale='Blues',
        range_color=[0, 1]
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Probability",
        yaxis_title="Question Type"
    )
    return fig


def main():
    # Header
    st.markdown("<h1 class='main-header'>🔍 Gujarati Question Type Classifier</h1>", 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'classifier' not in st.session_state:
        st.session_state['classifier'] = None
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    # Load model automatically
    if st.session_state['classifier'] is None:
        with st.spinner("Loading model..."):
            classifier, model_path = load_model()
            if classifier:
                st.session_state['classifier'] = classifier
                st.session_state['model_path'] = model_path
                st.success(f"✅ Model loaded: {model_path}")
            else:
                st.error("⚠️ No model file found. Please train the model first.")
                st.info("Run: python enhanced_train_question_classifier.py")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Dashboard")
        
        if st.session_state['classifier']:
            classifier = st.session_state['classifier']
            
            st.markdown("### Model Info")
            st.info(f"""
            **Model:** {st.session_state.get('model_path', 'Unknown')}
            **Classes:** {len(classifier.question_types)}
            """)
            
            st.markdown("### Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("History", len(st.session_state['history']))
            with col2:
                if st.session_state['history']:
                    types = [h['type'] for h in st.session_state['history']]
                    most_common = Counter(types).most_common(1)[0]
                    st.metric("Most Common", most_common[0])
            
            st.markdown("### Quick Actions")
            if st.button("🔄 Clear History", use_container_width=True):
                st.session_state['history'] = []
                st.success("History cleared!")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Single Question", "📚 Batch Analysis", "📖 Context Analysis"])
    
    # Tab 1: Single Question
    with tab1:
        st.markdown("<h2 class='sub-header'>Single Question Classification</h2>", 
                   unsafe_allow_html=True)
        
        if not st.session_state['classifier']:
            st.warning("Please wait for model to load...")
            return
        
        # Question input
        question = st.text_area(
            "Enter Gujarati question:",
            height=100,
            placeholder="e.g., અમદાવાદની સ્થાપના કોણે કરી?",
            key="single_q"
        )
        
        # Example buttons
        st.markdown("**Example Questions:**")
        examples = [
            "અમદાવાદની સ્થાપના કોણે કરી?",
            "ગીર રાષ્ટ્રીય ઉદ્યાનનો વિસ્તાર કેટલો છે?",
            "ગીરમાં જોવા મળતા પ્રાણીઓની યાદી બનાવો.",
            "એશિયાઇ સિંહ અને આફ્રિકન સિંહમાં શું તફાવત છે?",
            "જો ગીર ન રક્ષિત કરવામાં આવે તો શું થશે?",
            "આ ફકરાનો મુખ્ય વિષય શું છે?"
        ]
        
        cols = st.columns(3)
        for i, ex in enumerate(examples[:3]):
            with cols[i]:
                if st.button(f"Example {i+1}", key=f"ex1_{i}"):
                    st.session_state['single_q'] = ex
                    st.rerun()
        
        cols = st.columns(3)
        for i, ex in enumerate(examples[3:]):
            with cols[i]:
                if st.button(f"Example {i+4}", key=f"ex1_{i+3}"):
                    st.session_state['single_q'] = ex
                    st.rerun()
        
        # Classify button
        if st.button("🔍 Classify", type="primary", use_container_width=True):
            if question:
                with st.spinner("Analyzing..."):
                    classifier = st.session_state['classifier']
                    pred_type, confidence, top_3, prob_dict = classifier.predict(question)
                    
                    if pred_type:
                        # Add to history
                        st.session_state['history'].append({
                            'question': question[:30] + "..." if len(question) > 30 else question,
                            'type': pred_type,
                            'confidence': confidence,
                            'timestamp': time.strftime("%H:%M:%S")
                        })
                        
                        # Results
                        st.markdown("---")
                        
                        # Prediction box
                        color_class, conf_text, _ = get_confidence_info(confidence)
                        st.markdown(f"""
                        <div class='prediction-box'>
                            <h2 style='color: white;'>🎯 Prediction: {pred_type}</h2>
                            <h3 style='color: white;'>{conf_text}: {confidence:.4f}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Top 3 Predictions:**")
                            for i, (q_type, prob) in enumerate(top_3, 1):
                                st.progress(float(prob), text=f"{i}. {q_type}: {prob:.4f}")
                        
                        with col2:
                            fig = create_probability_chart(prob_dict)
                            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Batch Analysis
    with tab2:
        st.markdown("<h2 class='sub-header'>Batch Analysis</h2>", unsafe_allow_html=True)
        
        if not st.session_state['classifier']:
            st.warning("Please wait for model to load...")
            return
        
        questions_text = st.text_area(
            "Enter questions (one per line):",
            height=200,
            placeholder="Question 1\nQuestion 2\nQuestion 3\n..."
        )
        
        if st.button("📊 Analyze Batch", type="primary", use_container_width=True):
            if questions_text:
                questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
                
                with st.spinner(f"Processing {len(questions)} questions..."):
                    classifier = st.session_state['classifier']
                    results = []
                    
                    for q in questions:
                        pred_type, confidence, _, _ = classifier.predict(q)
                        results.append({
                            'Question': q[:30] + "..." if len(q) > 30 else q,
                            'Type': pred_type,
                            'Confidence': f"{confidence:.4f}"
                        })
                    
                    # Display results
                    st.markdown("---")
                    
                    df = pd.DataFrame(results)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        type_counts = df['Type'].value_counts()
                        fig = px.pie(
                            values=type_counts.values,
                            names=type_counts.index,
                            title="Type Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "batch_results.csv",
                        "text/csv",
                        use_container_width=True
                    )
    
    # Tab 3: Context Analysis
    with tab3:
        st.markdown("<h2 class='sub-header'>Context Analysis</h2>", unsafe_allow_html=True)
        
        if not st.session_state['classifier']:
            st.warning("Please wait for model to load...")
            return
        
        # Context input
        context = st.text_area(
            "Enter Gujarati context:",
            height=150,
            placeholder="Paste a paragraph here...",
            key="context"
        )
        
        # Questions input
        questions_input = st.text_area(
            "Enter questions from this context:",
            height=150,
            placeholder="Question 1\nQuestion 2\nQuestion 3\n...",
            key="context_qs"
        )
        
        if st.button("🔍 Analyze Context", type="primary", use_container_width=True):
            if context and questions_input:
                questions = [q.strip() for q in questions_input.split('\n') if q.strip()]
                
                with st.spinner(f"Analyzing {len(questions)} questions..."):
                    classifier = st.session_state['classifier']
                    results = []
                    
                    for q in questions:
                        pred_type, confidence, _, _ = classifier.predict(q)
                        results.append({
                            'Question': q[:30] + "..." if len(q) > 30 else q,
                            'Type': pred_type,
                            'Confidence': confidence
                        })
                    
                    # Display results
                    st.markdown("---")
                    
                    df = pd.DataFrame(results)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        type_counts = df['Type'].value_counts()
                        fig = px.pie(
                            values=type_counts.values,
                            names=type_counts.index,
                            title="Type Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.histogram(
                            df,
                            x='Confidence',
                            nbins=20,
                            title="Confidence Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "context_results.csv",
                        "text/csv",
                        use_container_width=True
                    )
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <p>🔍 Logistic Regression-Based Question Classification for Gujarati</p>
        <p>Made with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
