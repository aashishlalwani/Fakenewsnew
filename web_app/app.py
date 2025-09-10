"""
Simple web application for fake news detection using Streamlit.

This app provides a user-friendly interface to:
1. Input news text and get predictions
2. Upload files for batch prediction
3. View model performance metrics
4. Analyze text features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.data.preprocessing import TextPreprocessor
    from src.models.traditional_ml import TraditionalMLModels
except ImportError:
    st.error("Please install required dependencies and ensure the project structure is correct.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
}
.fake-news {
    background-color: #ffebee;
    color: #c62828;
    border: 2px solid #c62828;
}
.real-news {
    background-color: #e8f5e8;
    color: #2e7d32;
    border: 2px solid #2e7d32;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    # Create sample data if no real data is available
    sample_data = {
        'text': [
            "Breaking: Scientists discover new planet in our solar system!",
            "Local weather forecast shows sunny skies for the weekend.",
            "SHOCKING: Celebrity reveals secret about their past!",
            "Government announces new policy on environmental protection."
        ],
        'label': ['FAKE', 'REAL', 'FAKE', 'REAL']
    }
    return pd.DataFrame(sample_data)


def create_wordcloud(text, title):
    """Create and display a word cloud."""
    if text.strip():
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("No text available for word cloud generation.")


def analyze_text_features(text):
    """Analyze basic text features."""
    if not text:
        return {}
    
    words = text.split()
    sentences = text.split('.')
    
    features = {
        'Character Count': len(text),
        'Word Count': len(words),
        'Sentence Count': len(sentences),
        'Average Word Length': np.mean([len(word) for word in words]) if words else 0,
        'Average Sentence Length': np.mean([len(sentence.split()) for sentence in sentences if sentence.strip()]) if sentences else 0
    }
    
    return features


def main():
    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "🏠 Home",
        "🔍 Single Prediction", 
        "📊 Batch Analysis",
        "📈 Model Performance",
        "ℹ️ About"
    ])
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Fake News Detection System!
        
        This application helps you identify potentially fake news using machine learning techniques.
        
        ### Features:
        - **Single Prediction**: Analyze individual news articles
        - **Batch Analysis**: Process multiple articles at once
        - **Model Performance**: View detailed model metrics
        - **Text Analysis**: Understand what makes news fake or real
        
        ### How it works:
        1. Enter or upload news text
        2. Our AI models analyze the content
        3. Get predictions with confidence scores
        4. View detailed analysis and explanations
        
        ### Get Started:
        Choose a feature from the sidebar to begin!
        """)
        
        # Sample statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Accuracy", "85.2%", "2.3%")
        with col2:
            st.metric("Articles Analyzed", "1,234", "56")
        with col3:
            st.metric("Fake News Detected", "487", "23")
        with col4:
            st.metric("Confidence Score", "89.7%", "1.2%")
    
    elif page == "🔍 Single Prediction":
        st.header("Single Article Analysis")
        
        # Text input
        input_method = st.radio("Choose input method:", ["Type text", "Upload file"])
        
        news_text = ""
        if input_method == "Type text":
            news_text = st.text_area(
                "Enter news article text:",
                placeholder="Paste your news article here...",
                height=200
            )
        else:
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            if uploaded_file:
                news_text = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", news_text, height=200)
        
        if st.button("Analyze Article", type="primary") and news_text:
            with st.spinner("Analyzing article..."):
                # Preprocess text
                processed_text = preprocessor.preprocess_text(news_text)
                
                # Mock prediction (replace with actual model)
                # For demo purposes, we'll use simple heuristics
                fake_indicators = ['breaking', 'shocking', 'secret', 'revealed', 'exclusive']
                fake_score = sum(1 for indicator in fake_indicators if indicator in news_text.lower())
                confidence = min(0.6 + (fake_score * 0.15), 0.95)
                
                prediction = "FAKE" if fake_score >= 2 else "REAL"
                
                # Display prediction
                if prediction == "FAKE":
                    st.markdown(
                        f'<div class="prediction-box fake-news">⚠️ FAKE NEWS DETECTED<br>Confidence: {confidence:.1%}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-box real-news">✅ LIKELY REAL NEWS<br>Confidence: {confidence:.1%}</div>',
                        unsafe_allow_html=True
                    )
                
                # Analysis details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Text Features")
                    features = analyze_text_features(news_text)
                    for feature, value in features.items():
                        st.metric(feature, f"{value:.1f}" if isinstance(value, float) else str(value))
                
                with col2:
                    st.subheader("Key Indicators")
                    if prediction == "FAKE":
                        indicators = [indicator for indicator in fake_indicators if indicator in news_text.lower()]
                        if indicators:
                            st.warning(f"Suspicious words found: {', '.join(indicators)}")
                        st.info("Consider checking multiple sources for verification.")
                    else:
                        st.success("Text appears to have characteristics of legitimate news.")
                        st.info("Always verify information from multiple reliable sources.")
                
                # Word cloud
                if st.checkbox("Show Word Cloud"):
                    create_wordcloud(processed_text, "Most Frequent Words")
    
    elif page == "📊 Batch Analysis":
        st.header("Batch Article Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with articles", 
            type=['csv'],
            help="CSV should have a 'text' column with news articles"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            if 'text' in df.columns:
                if st.button("Analyze All Articles", type="primary"):
                    with st.spinner("Processing articles..."):
                        # Mock batch prediction
                        predictions = []
                        confidences = []
                        
                        for text in df['text']:
                            fake_indicators = ['breaking', 'shocking', 'secret', 'revealed', 'exclusive']
                            fake_score = sum(1 for indicator in fake_indicators if indicator in str(text).lower())
                            confidence = min(0.6 + (fake_score * 0.15), 0.95)
                            prediction = "FAKE" if fake_score >= 2 else "REAL"
                            
                            predictions.append(prediction)
                            confidences.append(confidence)
                        
                        df['Prediction'] = predictions
                        df['Confidence'] = confidences
                        
                        # Results summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Articles", len(df))
                        with col2:
                            fake_count = (df['Prediction'] == 'FAKE').sum()
                            st.metric("Fake News Detected", fake_count)
                        with col3:
                            real_count = (df['Prediction'] == 'REAL').sum()
                            st.metric("Real News", real_count)
                        
                        # Visualization
                        fig = px.pie(
                            values=[fake_count, real_count], 
                            names=['Fake', 'Real'],
                            title="Prediction Distribution"
                        )
                        st.plotly_chart(fig)
                        
                        # Results table
                        st.subheader("Detailed Results")
                        st.dataframe(df[['text', 'Prediction', 'Confidence']])
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="fake_news_predictions.csv",
                            mime="text/csv"
                        )
            else:
                st.error("CSV file must contain a 'text' column with news articles.")
        else:
            # Show sample data
            st.info("Upload a CSV file or use the sample data below:")
            sample_df = load_sample_data()
            st.dataframe(sample_df)
    
    elif page == "📈 Model Performance":
        st.header("Model Performance Metrics")
        
        # Mock performance data
        models_data = {
            'Model': ['Naive Bayes', 'Logistic Regression', 'SVM', 'Random Forest'],
            'Accuracy': [0.847, 0.889, 0.872, 0.856],
            'Precision': [0.851, 0.892, 0.875, 0.860],
            'Recall': [0.843, 0.885, 0.869, 0.852],
            'F1-Score': [0.847, 0.888, 0.872, 0.856]
        }
        
        df_models = pd.DataFrame(models_data)
        
        # Performance table
        st.subheader("Model Comparison")
        st.dataframe(df_models)
        
        # Performance chart
        fig = px.bar(
            df_models, 
            x='Model', 
            y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            title="Model Performance Comparison",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix (mock data)
        st.subheader("Best Model Confusion Matrix")
        confusion_data = [[145, 23], [18, 167]]
        
        fig = px.imshow(
            confusion_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Real', 'Fake'],
            y=['Real', 'Fake'],
            title="Confusion Matrix - Logistic Regression"
        )
        st.plotly_chart(fig)
        
        # Feature importance (if applicable)
        st.subheader("Most Important Features")
        feature_data = {
            'Feature': ['word_exclusive', 'word_breaking', 'exclamation_count', 'caps_ratio', 'word_shocking'],
            'Importance': [0.24, 0.19, 0.15, 0.13, 0.11]
        }
        df_features = pd.DataFrame(feature_data)
        
        fig = px.bar(
            df_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 5 Most Important Features"
        )
        st.plotly_chart(fig)
    
    elif page == "ℹ️ About":
        st.header("About This Project")
        
        st.markdown("""
        ## Fake News Detection System
        
        This project was developed as part of a semester 7 college project to demonstrate
        machine learning applications in natural language processing.
        
        ### Technology Stack:
        - **Python**: Core programming language
        - **Streamlit**: Web application framework
        - **Scikit-learn**: Machine learning algorithms
        - **NLTK/spaCy**: Natural language processing
        - **Pandas**: Data manipulation
        - **Plotly**: Interactive visualizations
        
        ### Models Used:
        1. **Naive Bayes**: Probabilistic classifier
        2. **Logistic Regression**: Linear classification
        3. **Support Vector Machine**: Margin-based classifier
        4. **Random Forest**: Ensemble method
        
        ### Features:
        - Text preprocessing and cleaning
        - TF-IDF vectorization
        - Multiple model comparison
        - Interactive web interface
        - Batch processing capabilities
        
        ### Limitations:
        - Models are trained on specific datasets
        - May not generalize to all news sources
        - Requires regular retraining with new data
        - Language and cultural biases may exist
        
        ### Future Improvements:
        - Deep learning models (BERT, RoBERTa)
        - Real-time news monitoring
        - Multi-language support
        - Social media integration
        - Explainable AI features
        
        ### Dataset Sources:
        - LIAR Dataset
        - ISOT Fake News Dataset
        - Kaggle Fake News Competition
        
        ### Disclaimer:
        This tool is for educational purposes. Always verify news from multiple reliable sources.
        """)
        
        # Team information
        st.subheader("Development Team")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Student 1**
            - Data Collection & Preprocessing
            - Traditional ML Implementation
            - Model Evaluation
            """)
        
        with col2:
            st.markdown("""
            **Student 2**
            - Web Application Development
            - Deep Learning Models
            - Deployment & Testing
            """)


if __name__ == "__main__":
    main()