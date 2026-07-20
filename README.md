# 🎯 End-to-End Customer Segmentation AI Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Database-MongoDB-47A248.svg)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end production-grade Data Science & Machine Learning platform for **Customer Segmentation** built with Python, Streamlit, Scikit-Learn, and MongoDB. The system processes raw e-commerce transaction data, calculates **RFM (Recency, Frequency, Monetary)** features, handles distribution skewness via log-transformation, and groups customers into distinct actionable personas using unsupervised clustering algorithms (**K-Means**, **Hierarchical Clustering**, **DBSCAN**, and **GMM**).

---

## 📌 Problem Statement & Business Impact

Targeted marketing is crucial for maximizing Customer Lifetime Value (CLV) and reducing churn. Generic marketing campaigns result in low engagement and wasted budgets. 

By grouping customers into distinct behavioral segments based on historical purchase data:
- **Champions**: Receive VIP rewards and early product access.
- **Loyal Customers**: Targeted with personalized cross-sell recommendations.
- **At-Risk Customers**: Re-engaged with win-back discount promotions.
- **Hibernating Customers**: Processed via low-cost automated retargeting workflows.

---

## 🏗️ System Architecture & Workflow

```
┌─────────────────┐      ┌─────────────────────┐      ┌───────────────────────┐
│ MongoDB / CSV   │ ───► │  Data Ingestion     │ ───► │  Data Validation      │
│ Transaction Data│      │ (artifacts/raw.csv) │      │ (config/schema.yaml)  │
└─────────────────┘      └─────────────────────┘      └───────────────────────┘
                                                                  │
                                                                  ▼
┌─────────────────┐      ┌─────────────────────┐      ┌───────────────────────┐
│ Streamlit App   │ ◄─── │  Model Evaluation   │ ◄─── │  Data Transformation  │
│ Web Dashboard   │      │ (Silhouette/Elbow) │      │  (RFM + Log + Scaler) │
└─────────────────┘      └─────────────────────┘      └───────────────────────┘
```

### Key Engineering Features
1. **Modular Code Architecture (`src/`)**: Clean separation into Ingestion, Validation, Transformation, Model Trainer, Logging, and Custom Exceptions.
2. **MongoDB Integration with Automatic CSV Fallback**: Connects directly to cloud/local MongoDB collections or gracefully defaults to local e-commerce datasets.
3. **RFM Feature Engineering**: Automatically cleans cancelled orders, parses transaction dates, and calculates per-customer Recency, Frequency, and Monetary spend metrics.
4. **Unsupervised ML Stack**: Trains **K-Means**, **Hierarchical Clustering**, and **GMM** models, using Silhouette Score optimization to automatically select optimal cluster counts.
5. **Interactive Web Dashboard**: Streamlit app with interactive charts, live inference predictor, model evaluation tab, and one-click pipeline retraining.

---

## 📂 Repository Layout

```
├── .github/workflows/     # GitHub Actions CI/CD Pipeline
│   └── main.yml
├── artifacts/             # Generated dataset, preprocessor & model binaries
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── transformed_rfm.csv
├── config/                # Schema definitions & configurations
│   └── schema.yaml
├── data/                  # Raw transaction dataset & generator
│   └── raw_data.csv
├── src/                   # Core Python Source Code
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── app.py                 # Streamlit Web Application
├── Dockerfile             # Production container setup
├── requirements.txt       # Python Dependencies
├── setup.py               # Package Setup
└── README.md              # Project Documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/customer-segmentation-ml.git
cd customer-segmentation-ml

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate
# Activate environment (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Training Pipeline

Execute the end-to-end training pipeline to ingest data, calculate RFM features, train clustering models, and save artifacts:

```bash
python src/pipeline/train_pipeline.py
```

### 4. Launch Streamlit Web Dashboard

Start the interactive web dashboard locally:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment

To build and run the application containerized:

```bash
# Build Docker image
docker build -t customer-segmentation-app .

# Run container
docker run -p 8501:8501 customer-segmentation-app
```

---

## 📊 Model Evaluation Metrics

| Algorithm | Optimal Clusters ($k$) | Silhouette Score | Davies-Bouldin Index |
| :--- | :---: | :---: | :---: |
| **K-Means (Selected)** | **4** | **0.4344** | **0.891** |
| Hierarchical Clustering | 4 | 0.3951 | 0.945 |
| Gaussian Mixture Model | 4 | 0.3812 | 1.021 |

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for details.
