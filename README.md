# Sleep_Disorder_Detection
Overview
-This project is a machine learning web application that predicts the likelihood of a sleep disorder based on health and lifestyle data. -Users can input information such as age, sleep duration, stress level, heart rate, daily steps, gender, BMI, occupation, and blood pressure to receive a real-time prediction.

Features
- Predicts binary sleep disorder outcome: `No Sleep Disorder` or `Sleep Disorder`.
- Provides confidence score for the prediction.
- Interactive web interface using Gradio.
- Handles missing data and categorical features automatically.

Dataset
The model was trained on a dataset containing health and lifestyle features related to sleep disorders. Features include:
- Age, Sleep Duration, Quality of Sleep, Physical Activity Level, Stress Level, Heart Rate, Daily Steps
- Gender, Occupation, BMI Category, BP Systolic, BP Diastolic

Model
- Baseline: Logistic Regression
- Improved: SMOTE for class balancing + XGBoost
- Evaluation Metrics (Improved Model):
  - Accuracy: 96%
  - ROC AUC: 0.959
  - Precision, Recall, F1-Score improved for both classes


