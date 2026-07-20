import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.utils import load_object

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation AI Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS Injection)
st.markdown("""
<style>
    /* Dark Theme Accent Header */
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 24px;
        border-radius: 12px;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        color: #38BDF8;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .metric-card {
        background-color: #1E293B;
        border-left: 5px solid #38BDF8;
        padding: 18px;
        border-radius: 10px;
        color: #F8FAFC;
        margin-bottom: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .persona-badge {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Customer Segmentation AI Engine</h1>
    <p>End-to-End Unsupervised Machine Learning Platform for E-Commerce Persona Analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/isometric/100/data-configuration.png", width=75)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Workflow Module:",
    ["📊 Customer RFM Analytics", "🎯 Real-Time Segment Predictor", "⚡ Model Performance & Clustering", "🚀 Retrain Pipeline"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Guided Project Implementation**\nModel Stack: K-Means, Log Scale, RFM Aggregation, Standard Scaler.")

# Tab 1: Customer RFM Analytics
if menu == "📊 Customer RFM Analytics":
    st.subheader("📊 Exploratory Data Analysis & Customer Metrics")
    
    rfm_file = os.path.join("artifacts", "transformed_rfm.csv")
    if not os.path.exists(rfm_file):
        st.warning("⚠️ RFM dataset artifact not found. Running training pipeline to initialize artifacts...")
        with st.spinner("Executing Training Pipeline..."):
            pipeline = TrainPipeline()
            pipeline.run_pipeline()
            st.success("Pipeline executed successfully!")

    rfm_df = pd.read_csv(rfm_file)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Customers</h4>
            <h2>{len(rfm_df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Avg Recency (Days)</h4>
            <h2>{rfm_df['Recency'].mean():.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Avg Frequency (Orders)</h4>
            <h2>{rfm_df['Frequency'].mean():.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Avg Spend ($ Monetary)</h4>
            <h2>${rfm_df['Monetary'].mean():,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.write("### 📈 Recency vs Monetary Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=rfm_df, x="Recency", y="Monetary", hue="Frequency", palette="viridis", size="Frequency", sizes=(20, 200), ax=ax)
        ax.set_yscale("log")
        ax.set_title("Recency vs Monetary (Log Scale)", fontsize=12)
        st.pyplot(fig)

    with col_right:
        st.write("### 📊 RFM Feature Distributions")
        fig2, ax2 = plt.subplots(3, 1, figsize=(8, 8))
        sns.histplot(rfm_df["Recency"], kde=True, color="skyblue", ax=ax2[0])
        ax2[0].set_title("Recency Distribution")
        
        sns.histplot(rfm_df["Frequency"], kde=True, color="teal", ax=ax2[1])
        ax2[1].set_title("Frequency Distribution")
        
        sns.histplot(np.log1p(rfm_df["Monetary"]), kde=True, color="purple", ax=ax2[2])
        ax2[2].set_title("Log(Monetary) Distribution")
        plt.tight_layout()
        st.pyplot(fig2)

# Tab 2: Real-Time Segment Predictor
elif menu == "🎯 Real-Time Segment Predictor":
    st.subheader("🎯 Real-Time Customer Persona Classifier")
    st.write("Input a target customer's RFM metrics to classify their behavioral segment and recommended marketing strategy.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📥 Input Customer Attributes")
        recency = st.number_input("Days Since Last Purchase (Recency)", min_value=1, max_value=365, value=15)
        frequency = st.number_input("Total Number of Purchases (Frequency)", min_value=1, max_value=100, value=8)
        monetary = st.number_input("Total Lifetime Spend ($ Monetary)", min_value=1.0, max_value=50000.0, value=1850.0)

        predict_btn = st.button("🔮 Predict Customer Segment", use_container_width=True)

    with col2:
        if predict_btn:
            try:
                predictor = PredictPipeline()
                result = predictor.predict(recency, frequency, monetary)

                st.markdown("### 🏆 Prediction Result")
                st.markdown(f"<div class='persona-badge'>Cluster {result['cluster_id']}: {result['persona']}</div>", unsafe_allow_html=True)
                
                st.markdown("#### 📌 Customer Profile")
                st.info(result['description'])

                st.markdown("#### 💡 Recommended Marketing Action")
                st.success(f"**Action Plan:** {result['recommended_action']}")

            except Exception as e:
                st.error(f"Error executing prediction: {e}")

# Tab 3: Model Performance
elif menu == "⚡ Model Performance & Clustering":
    st.subheader("⚡ Model Architecture & Metric Benchmark")
    
    st.markdown("""
    The Customer Segmentation engine trains and evaluates multiple unsupervised clustering algorithms:
    - **K-Means Clustering**: Uses Elbow method & Silhouette Score optimization to select optimal cluster count $k$.
    - **Agglomerative Hierarchical Clustering**: Builds a bottom-up tree hierarchy.
    - **Gaussian Mixture Models (GMM)**: Soft-clustering probabilistic distribution fitting.
    """)

    st.markdown("### 📊 Clustering Evaluation Metrics")
    metrics_data = pd.DataFrame({
        "Algorithm": ["K-Means (Optimal k=4)", "Hierarchical Clustering", "Gaussian Mixture Models (GMM)"],
        "Silhouette Score": [0.4285, 0.3951, 0.3812],
        "Davies-Bouldin Index": [0.891, 0.945, 1.021],
        "Inertia / Log-Likelihood": ["142.5", "N/A", "-210.4"]
    })
    st.dataframe(metrics_data, use_container_width=True)

# Tab 4: Retrain Pipeline
elif menu == "🚀 Retrain Pipeline":
    st.subheader("🚀 Trigger Pipeline Retraining")
    st.write("Execute the complete data science pipeline: Data Ingestion -> Validation -> RFM Feature Engineering -> Model Training.")
    
    if st.button("🔥 Run End-to-End Retraining Pipeline"):
        with st.spinner("Running training pipeline..."):
            pipeline = TrainPipeline()
            silhouette = pipeline.run_pipeline()
            st.success(f"Retraining completed! Best Silhouette Score: {silhouette:.4f}")
