# app_final.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import time
from collections import Counter
from scipy.sparse import hstack, csr_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    /* Main header styling */
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
    
    /* Sub-header styling */
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    
    /* Prediction box styling */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Confidence levels */
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
    
    /* Metric cards */
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
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #6B7280;
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.9rem;
        border-top: 1px solid #E5E7EB;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    
    /* Success message styling */
    .success-message {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)


class GujaratiQuestionClassifier:
    """Enhanced classifier with linguistic features"""
    
    def __init__(self, model_data):
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.count_vectorizer = model_data.get('count_vectorizer', None)
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.question_types = model_data['question_types']
        
    def extract_linguistic_features(self, questions):
        """Extract linguistic features from Gujarati questions"""
        features = []
        
        # Gujarati question words and markers
        gujarati_q_words = {
            'કોણ': 0, 'શું': 1, 'ક્યાં': 2, 'ક્યારે': 3, 
            'કેમ': 4, 'કેવી': 5, 'કેવું': 6, 'કેટલા': 7,
            'કેટલી': 8, 'કેટલો': 9, 'શા': 10, 'શાનો': 11,
            'કયા': 12, 'કઈ': 13
        }
        
        comparative_markers = ['અને', 'વચ્ચે', 'સામ્યતા', 'તફાવત', 'સરખામણી']
        predictive_markers = ['જો', 'તો', 'થાત', 'થશે', 'ભવિષ્ય']
        evaluative_markers = ['તમારા', 'મતે', 'સૌથી', 'મહત્વ', 'શ્રેષ્ઠ']
        list_markers = ['યાદી', 'બનાવો', 'ચાર', 'ત્રણ', 'પાંચ', 'કઈ', 'કેટલા']
        definition_markers = ['શબ્દનો', 'અર્થ', 'એટલે', 'શું છે', 'કોને કહેવાય']
        numerical_markers = ['કેટલા', 'કેટલી', 'કેટલો', 'ક્યારે', 'વર્ષ', 'તારીખ', 'સમય']
        factual_markers = ['કોણ', 'શું', 'ક્યાં', 'ક્યારે']
        inferential_markers = ['કેમ', 'શા માટે', 'કારણ']
        thematic_markers = ['મુખ્ય', 'વિષય', 'ફકરો', 'સંદેશ']
        
        for q in questions:
            q_features = []
            
            # Basic features
            q_features.append(len(q))  # Length
            q_features.append(len(q.split()))  # Word count
            
            # Question words presence (one-hot like)
            for word in gujarati_q_words:
                q_features.append(1 if word in q else 0)
            
            # Marker counts
            q_features.append(sum(1 for marker in comparative_markers if marker in q))
            q_features.append(sum(1 for marker in predictive_markers if marker in q))
            q_features.append(sum(1 for marker in evaluative_markers if marker in q))
            q_features.append(sum(1 for marker in list_markers if marker in q))
            q_features.append(sum(1 for marker in definition_markers if marker in q))
            q_features.append(sum(1 for marker in numerical_markers if marker in q))
            q_features.append(sum(1 for marker in factual_markers if marker in q))
            q_features.append(sum(1 for marker in inferential_markers if marker in q))
            q_features.append(sum(1 for marker in thematic_markers if marker in q))
            
            features.append(q_features)
        
        return np.array(features)
    
    def predict(self, question):
        """Predict question type with confidence scores"""
        try:
            # Extract features
            tfidf_features = self.tfidf_vectorizer.transform([question])
            
            if self.count_vectorizer:
                count_features = self.count_vectorizer.transform([question])
                ling_features = self.extract_linguistic_features([question])
                ling_features_sparse = csr_matrix(ling_features)
                combined_features = hstack([tfidf_features, count_features, ling_features_sparse])
            else:
                combined_features = tfidf_features
            
            # Predict
            pred_encoded = self.classifier.predict(combined_features)[0]
            pred_type = self.label_encoder.inverse_transform([pred_encoded])[0]
            
            # Get probabilities
            probs = self.classifier.predict_proba(combined_features)[0]
            confidence = float(max(probs))
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probs)[-3:][::-1]
            top_3 = [(self.label_encoder.inverse_transform([i])[0], float(probs[i])) 
                     for i in top_3_indices]
            
            # Get all probabilities as dictionary
            prob_dict = {
                self.label_encoder.inverse_transform([i])[0]: float(probs[i])
                for i in range(len(self.question_types))
            }
            
            return pred_type, confidence, top_3, prob_dict
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, 0.0, [], {}


@st.cache_resource
def load_model(model_path):
    """Load the trained model with caching"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return GujaratiQuestionClassifier(model_data)
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def get_confidence_info(confidence):
    """Get confidence level styling and label"""
    if confidence >= 0.8:
        return "confidence-high", "🟢 High Confidence", "#10B981"
    elif confidence >= 0.6:
        return "confidence-medium", "🟡 Medium Confidence", "#F59E0B"
    else:
        return "confidence-low", "🔴 Low Confidence", "#EF4444"


def create_probability_chart(prob_dict, title="Class Probabilities"):
    """Create a bar chart of probabilities"""
    df = pd.DataFrame({
        'Type': list(prob_dict.keys()),
        'Probability': list(prob_dict.values())
    }).sort_values('Probability', ascending=True)
    
    fig = px.bar(
        df.tail(10),
        x='Probability',
        y='Type',
        orientation='h',
        title=title,
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
    if 'current_model' not in st.session_state:
        st.session_state['current_model'] = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Model Control Panel")
        
        # Model selection
        model_options = {
            "Enhanced Model (Recommended)": "enhanced_gujarati_question_classifier.pkl",
            "Basic Model": "gujarati_question_classifier.pkl"
        }
        
        selected_model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0
        )
        
        model_path = model_options[selected_model_name]
        
        # Load model button
        if st.button("🚀 Load Model", use_container_width=True):
            with st.spinner(f"Loading {selected_model_name}..."):
                classifier = load_model(model_path)
                if classifier:
                    st.session_state['classifier'] = classifier
                    st.session_state['current_model'] = selected_model_name
                    st.success(f"✅ {selected_model_name} loaded!")
                    st.balloons()
        
        st.markdown("---")
        
        # Model info (if loaded)
        if st.session_state['classifier']:
            st.markdown("### 📈 Model Statistics")
            classifier = st.session_state['classifier']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Classes", len(classifier.question_types))
            with col2:
                st.metric("History", len(st.session_state['history']))
            
            st.markdown("**Supported Types:**")
            for q_type in classifier.question_types[:5]:
                st.markdown(f"- {q_type}")
            if len(classifier.question_types) > 5:
                st.markdown(f"- *...and {len(classifier.question_types)-5} more*")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Clear History", use_container_width=True):
            st.session_state['history'] = []
            st.success("History cleared!")
        
        if st.button("📊 Export History", use_container_width=True):
            if st.session_state['history']:
                history_df = pd.DataFrame(st.session_state['history'])
                csv = history_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    csv,
                    "classification_history.csv",
                    "text/csv"
                )
    
    # Main content area - Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Single Question", 
        "📚 Batch Analysis", 
        "📖 Context Analysis",
        "📊 Analytics"
    ])
    
    # Tab 1: Single Question
    with tab1:
        st.markdown("<h2 class='sub-header'>Single Question Classification</h2>", 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            question = st.text_area(
                "Enter Gujarati question:",
                height=120,
                placeholder="e.g., અમદાવાદની સ્થાપના કોણે કરી?",
                key="single_q"
            )
        
        with col2:
            st.markdown("**📋 Example Questions:**")
            examples = [
                "અમદાવાદની સ્થાપના કોણે કરી?",
                "ગીરમાં કેટલા એશિયાઇ સિંહો છે?",
                "ગીરમાં જોવા મળતા પ્રાણીઓની યાદી બનાવો.",
                "એશિયાઇ અને આફ્રિકન સિંહમાં શું તફાવત છે?",
                "જો ગીર ન રક્ષિત કરવામાં આવે તો શું થશે?"
            ]
            for i, ex in enumerate(examples):
                if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
                    st.session_state['single_q'] = ex
                    st.rerun()
        
        # Classify button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
        
        if classify_btn and question:
            if not st.session_state['classifier']:
                st.error("⚠️ Please load a model first!")
            else:
                with st.spinner("Analyzing question..."):
                    classifier = st.session_state['classifier']
                    pred_type, confidence, top_3, prob_dict = classifier.predict(question)
                    
                    if pred_type:
                        # Add to history
                        st.session_state['history'].append({
                            'question': question[:50] + "..." if len(question) > 50 else question,
                            'type': pred_type,
                            'confidence': confidence,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        # Display results
                        st.markdown("---")
                        
                        # Prediction card
                        color_class, conf_text, conf_color = get_confidence_info(confidence)
                        st.markdown(f"""
                        <div class='prediction-box'>
                            <h2 style='color: white; margin-bottom: 1rem;'>🎯 Prediction Result</h2>
                            <h1 style='color: white; font-size: 2.5rem; margin-bottom: 1rem;'>{pred_type}</h1>
                            <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 0.5rem;'>
                                <h3 style='color: white;'>{conf_text}: {confidence:.4f}</h3>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📊 Top 3 Predictions:**")
                            for i, (q_type, prob) in enumerate(top_3, 1):
                                st.progress(float(prob), text=f"{i}. {q_type}: {prob:.4f}")
                        
                        with col2:
                            # Probability chart
                            fig = create_probability_chart(prob_dict)
                            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Batch Analysis
    with tab2:
        st.markdown("<h2 class='sub-header'>Batch Question Analysis</h2>", unsafe_allow_html=True)
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📝 Text Input", "📁 Upload File"],
            horizontal=True
        )
        
        if input_method == "📝 Text Input":
            questions_text = st.text_area(
                "Enter questions (one per line):",
                height=200,
                placeholder="Question 1\nQuestion 2\nQuestion 3\n..."
            )
            
            if st.button("📊 Analyze Batch", type="primary", use_container_width=True):
                if not st.session_state['classifier']:
                    st.error("⚠️ Please load a model first!")
                elif questions_text:
                    questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
                    process_batch(questions, st.session_state['classifier'])
        
        else:  # File upload
            uploaded_file = st.file_uploader(
                "Upload a text file (one question per line)",
                type=['txt', 'csv']
            )
            
            if uploaded_file and st.button("📊 Process File", type="primary", use_container_width=True):
                if not st.session_state['classifier']:
                    st.error("⚠️ Please load a model first!")
                else:
                    content = uploaded_file.getvalue().decode('utf-8')
                    questions = [q.strip() for q in content.split('\n') if q.strip()]
                    process_batch(questions, st.session_state['classifier'])
    
    # Tab 3: Context Analysis
    with tab3:
        st.markdown("<h2 class='sub-header'>Context & Questions Analysis</h2>", unsafe_allow_html=True)
        
        # Context input
        context = st.text_area(
            "📖 Enter Gujarati context/paragraph:",
            height=150,
            placeholder="Paste a paragraph here...",
            key="context_input"
        )
        
        # Questions input
        questions_input = st.text_area(
            "❓ Enter questions from this context (one per line):",
            height=150,
            placeholder="Question 1\nQuestion 2\nQuestion 3\n...",
            key="context_questions"
        )
        
        if st.button("🔍 Analyze Context", type="primary", use_container_width=True):
            if not st.session_state['classifier']:
                st.error("⚠️ Please load a model first!")
            elif context and questions_input:
                questions = [q.strip() for q in questions_input.split('\n') if q.strip()]
                classifier = st.session_state['classifier']
                
                with st.spinner(f"Analyzing {len(questions)} questions..."):
                    results = []
                    for q in questions:
                        pred_type, confidence, _, _ = classifier.predict(q)
                        results.append({
                            'Question': q[:50] + "..." if len(q) > 50 else q,
                            'Type': pred_type,
                            'Confidence': confidence
                        })
                    
                    # Display results
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        df = pd.DataFrame(results)
                        type_counts = df['Type'].value_counts()
                        
                        fig = px.pie(
                            values=type_counts.values,
                            names=type_counts.index,
                            title="Question Type Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.histogram(
                            df,
                            x='Confidence',
                            nbins=20,
                            title="Confidence Distribution",
                            color_discrete_sequence=['#2563EB']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.markdown("**📋 Detailed Results:**")
                    display_df = df.copy()
                    display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.4f}")
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "context_analysis.csv",
                        "text/csv",
                        use_container_width=True
                    )
    
    # Tab 4: Analytics
    with tab4:
        st.markdown("<h2 class='sub-header'>Analytics Dashboard</h2>", unsafe_allow_html=True)
        
        if not st.session_state['classifier']:
            st.warning("Please load a model to view analytics.")
        else:
            classifier = st.session_state['classifier']
            
            # Model performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class='metric-card'>
                    <h3>Model Type</h3>
                    <p style='font-size: 1.2rem; font-weight: bold;'>Logistic Regression</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='metric-card'>
                    <h3>Accuracy</h3>
                    <p style='font-size: 1.2rem; font-weight: bold; color: #10B981;'>59.05%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class='metric-card'>
                    <h3>Classes</h3>
                    <p style='font-size: 1.2rem; font-weight: bold;'>9 Types</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class='metric-card'>
                    <h3>Training Samples</h3>
                    <p style='font-size: 1.2rem; font-weight: bold;'>523</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Question type guide
            st.markdown("### 📝 Question Type Guide")
            
            type_guide = pd.DataFrame({
                'Type': ['factual', 'numerical', 'list', 'comparative', 'definition', 
                        'inferential', 'predictive', 'evaluative', 'thematic'],
                'Description': [
                    'Asks for specific facts (who, what, where)',
                    'Asks for numbers, dates, quantities',
                    'Requests a list of items',
                    'Compares two or more things',
                    'Asks for meaning/definition of terms',
                    'Requires inference/reasoning',
                    'Asks about future/hypothetical scenarios',
                    'Asks for opinion/judgment',
                    'Asks about main theme/topic'
                ],
                'Example': [
                    'અમદાવાદની સ્થાપના કોણે કરી?',
                    'ગીરનો વિસ્તાર કેટલો છે?',
                    'ત્રણ ઐતિહાસિક ઇમારતોની યાદી બનાવો',
                    'અમદાવાદ અને મુંબઈમાં શું તફાવત છે?',
                    "'ગરબો' શબ્દનો અર્થ શું છે?",
                    'ગીરનું રાષ્ટ્રીય ઉદ્યાન બનવું શા માટે મહત્વપૂર્ણ?',
                    'જો ગીર ન રક્ષિત કરવામાં આવે તો શું થશે?',
                    'સૌથી મહત્વની ઓળખ તમારા મતે કઈ છે?',
                    'આ ફકરાનો મુખ્ય વિષય શું છે?'
                ]
            })
            
            st.dataframe(type_guide, use_container_width=True)
            
            # History analysis
            if st.session_state['history']:
                st.markdown("---")
                st.markdown("### 📊 Classification History")
                
                history_df = pd.DataFrame(st.session_state['history'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    type_counts = history_df['type'].value_counts()
                    fig = px.bar(
                        x=type_counts.index,
                        y=type_counts.values,
                        title="Question Types in History",
                        labels={'x': 'Type', 'y': 'Count'},
                        color=type_counts.values,
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(
                        history_df,
                        y='confidence',
                        title="Confidence Distribution",
                        points="all"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <p>🔍 Logistic Regression-Based Question Type Classification for Gujarati</p>
        <p>📊 Model Accuracy: 59.05% | 🎯 9 Question Types | 📝 523 Training Samples</p>
        <p>Made with ❤️ using Streamlit and Scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)


def process_batch(questions, classifier):
    """Process batch questions and display results"""
    results = []
    
    with st.spinner(f"Processing {len(questions)} questions..."):
        for q in questions:
            pred_type, confidence, _, _ = classifier.predict(q)
            results.append({
                'Question': q[:50] + "..." if len(q) > 50 else q,
                'Type': pred_type,
                'Confidence': confidence
            })
    
    # Display results
    st.markdown("---")
    st.markdown("### 📊 Batch Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        df = pd.DataFrame(results)
        
        # Summary stats
        st.metric("Total Questions", len(results))
        st.metric("Unique Types", df['Type'].nunique())
        
        # Type distribution
        type_counts = df['Type'].value_counts()
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence distribution
        fig = px.histogram(
            df,
            x='Confidence',
            nbins=20,
            title="Confidence Distribution",
            color_discrete_sequence=['#2563EB']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    st.markdown("**📋 Detailed Results:**")
    display_df = df.copy()
    display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Results",
        csv,
        "batch_results.csv",
        "text/csv",
        use_container_width=True
    )


if __name__ == "__main__":
    main()
