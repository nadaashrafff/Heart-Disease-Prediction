## Heart Disease Prediction Project

This project applies machine learning to predict the likelihood of heart disease using clinical data. It covers data preprocessing, feature engineering, supervised & unsupervised learning, model optimization, and deployment via Streamlit.

# Project Structure
Heart_Disease_Project/
│── data/                  
│   ├── heart_disease_uci.csv
│   ├── cleaned_heart.csv
│   ├── heart_selected_features.csv
│   ├── heart_pca.csv
│
│── notebooks/             
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│
│── models/                
│   ├── final_model.pkl
│   ├── model_pipeline.pkl
│   ├── model_metadata.json
│
│── results/               
│   ├── evaluation_metrics.txt
│
│── ui/                    
│   ├── app.py
│
│── requirements.txt       
│── README.md             
│── .gitignore             

# Installation

Clone the repo and create a virtual environment:

git clone https://github.com/your-username/Heart_Disease_Project.git
cd Heart_Disease_Project

# Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Workflow
1. Data Preprocessing

Cleaned raw UCI dataset

Handled missing values

Encoded categorical variables

Scaled numeric features

2. PCA (Dimensionality Reduction)

Reduced correlated features

Visualized variance explained

3. Feature Selection

Identified most predictive clinical features

Final set included:
age, sex_Male, trestbps, chol, thalach, exang, ca, oldpeak, cp_* types, thal_* types

4. Supervised Learning

Trained models: Logistic Regression, SVM, Random Forest, Decision Tree

Evaluated with Accuracy, Precision, Recall, F1, ROC-AUC

Best models:

Logistic Regression → ROC-AUC ≈ 0.90

SVM → ROC-AUC ≈ 0.90

5. Unsupervised Learning

K-Means and Hierarchical clustering

Compared clusters vs. actual labels

Moderate alignment observed

6. Hyperparameter Tuning

GridSearchCV on Logistic Regression & SVM

Final ROC-AUC ≈ 0.91

7. Model Export & Deployment

Saved as reproducible pipeline: preprocessing + model

Exported:

model_pipeline.pkl

model_metadata.json

8. Streamlit Web App

Simple UI for user input (age, sex, cholesterol, etc.)

Real-time prediction (disease vs no-disease)

Adjustable decision threshold

Run with:

streamlit run ui/app.py

Example UI

The app lets users enter clinical values and get instant prediction:

Input: Age, Blood Pressure, Cholesterol, Chest Pain, Thal, etc.

Output: Probability of Heart Disease

Interactive threshold adjustment

# Deployment

Local deployment: streamlit run ui/app.py

Bonus (optional): public access via Ngrok

# Requirements

Main libraries:

pandas, numpy

scikit-learn

matplotlib, seaborn

streamlit

joblib

Install with:

pip install -r requirements.txt

# Deliverables

Cleaned dataset with selected features

PCA results

Supervised and Unsupervised models

Optimized model with hyperparameters

Exported pipeline (.pkl) + metadata

Streamlit UI for predictions

GitHub repo with documentation
