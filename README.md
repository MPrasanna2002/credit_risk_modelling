# credit_risk_modelling
A comprehensive machine learning pipeline for credit risk assessment. This project explores data preprocessing, feature engineering, model training, and evaluation using classification algorithms like Logistic Regression, Random Forest, and XGBoost to predict the likelihood of loan default.

# Credit Risk Modelling using Machine Learning

This project builds a machine learning pipeline to predict credit risk — the likelihood of a loan applicant defaulting — using a classification model trained on historical data.

## Files

- `credit_risk_dataset (1).csv`: Dataset used for training and evaluation.
- `rf_model (2).pkl`: Trained Random Forest model.
- `scaler.pkl`: Pre-fitted data scaler used for preprocessing.

## Project Workflow

1. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Scale numerical features

2. **Exploratory Data Analysis (EDA)**
   - Distribution analysis
   - Correlation heatmaps
   - Class imbalance check

3. **Model Training**
   - Algorithms: Random Forest (primary), others optional
   - Train-test split
   - Hyperparameter tuning

4. **Model Evaluation**
   - Confusion Matrix
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC Score

5. **Model Serialization**
   - Trained model saved as `rf_model (2).pkl`
   - Scaler saved as `scaler.pkl`

Example packages used:

pandas

scikit-learn

matplotlib

seaborn

How to Use
Load the scaler and model:

import joblib

scaler = joblib.load("scaler.pkl")
model = joblib.load("rf_model (2).pkl")
Preprocess new input data using the same pipeline.

Make predictions:

prediction = model.predict(scaled_input)
