# Employee Attrition Prediction System

## Overview

This project is a Machine Learning-based prediction engine designed to identify whether an employee is likely to leave an organization (attrition) based on behavioral, financial, and work-related features.

It implements a complete ML pipeline:
**data preprocessing → feature engineering → model training → evaluation → prediction**

---

## Key Features

* Data cleaning and preprocessing pipeline
* Categorical encoding (One-Hot Encoding, Label Encoding)
* Feature scaling using StandardScaler
* Logistic Regression model training
* Model evaluation:

  * Accuracy Score
  * Classification Report
  * Confusion Matrix
* Interactive CLI-based prediction system

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* hvplot

---

## Project Structure

```
employee-attrition-prediction/
│
├── notebooks/        # EDA and experimentation
├── src/              # Core ML logic
├── data/             # Dataset (CSV included)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shy4m-003/Employee-Attrition-Prediction.git
cd employee-attrition-prediction
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run the prediction script

```bash
python src/predict.py
```

### Example Input

You will be prompted to enter:

* Age
* Monthly Income
* Distance From Home
* Job Satisfaction (1–4)
* Overtime (1 = Yes, 0 = No)

### Example Output

```
Prediction: The employee is likely to leave (Yes/No)
```

---

## Model Details

* Algorithm: Logistic Regression
* Train-Test Split: 80/20
* Feature Scaling: StandardScaler

### Evaluation Metrics

* Accuracy Score
* Classification Report
* Confusion Matrix

---

## Dataset

The dataset is included in the `data/` directory.

It contains features such as:

* Age
* Salary
* Job Role
* Work-Life Balance
* Satisfaction Metrics

---

## Future Enhancements

* Expose prediction as REST API (Spring Boot / FastAPI)
* Build web dashboard (Angular / React)
* Add advanced models (Random Forest, XGBoost)
* Real-time prediction pipeline
* Model persistence (save/load trained model)

---

## Product Vision

This system can act as a **core prediction engine** for:

* HR Analytics Dashboards
* Employee Retention Platforms
* Workforce Risk Monitoring Systems
* AI-driven HR Decision Tools

### Example Use Case

A company portal where HR inputs employee details →
system predicts attrition risk →
triggers retention strategies (alerts, interventions, incentives)

---

## Extensibility

This project can be extended into a full product by:

* Wrapping the model into a backend service (Spring Boot / FastAPI)
* Integrating with databases (employee records)
* Adding role-based dashboards (HR/Admin)
* Connecting to real-time company data pipelines

---

## Author

Shyam
