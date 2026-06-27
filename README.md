# Customer Churn Prediction using Machine Learning

Predicting customer attrition using machine learning classification models to support proactive customer retention strategies.

---

## 📌 Project Overview

Customer churn is one of the major challenges faced by subscription-based businesses. This project develops a machine learning model to predict whether a customer is likely to discontinue a service based on customer demographics, account information, and subscription details. Early identification of potential churn enables businesses to implement targeted retention strategies.

---

## 🎯 Problem Statement

Acquiring new customers is significantly more expensive than retaining existing ones. The objective of this project is to build an accurate machine learning model capable of predicting customer churn, enabling businesses to identify high-risk customers and improve customer retention.

---

## 📂 Dataset

- **Dataset:** IBM Telco Customer Churn Dataset
- **Source:** IBM Sample Dataset (Available on Kaggle)
- **Records:** 7,043 customers
- **Features:** 20 input features + 1 target variable
- **Target Variable:** Churn (Yes/No)

### Key Features

- Contract Type
- Tenure
- Monthly Charges
- Total Charges
- Internet Service
- Online Security
- Tech Support
- Payment Method

---

## ⚙️ Project Workflow

1. Data Loading and Cleaning
2. Exploratory Data Analysis (EDA)
3. Missing Value Handling
4. Label Encoding of Categorical Variables
5. Train-Test Split
6. Handling Class Imbalance using SMOTE
7. Model Training
8. Hyperparameter Tuning using GridSearchCV
9. Model Evaluation
10. Feature Importance Analysis
11. Model Serialization using Pickle

---

## 🤖 Machine Learning Models

- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

---

## 📊 Model Performance

### Cross Validation Accuracy

| Model | Accuracy |
|--------|----------|
| Decision Tree | 78% |
| Random Forest | **84%** |
| XGBoost | 83% |

### Final Tuned Random Forest Performance

| Metric | Score |
|--------|-------|
| Accuracy | **77.86%** |
| Precision | **0.57** |
| Recall | **0.64** |
| F1-Score | **0.60** |

---

## 🔍 Feature Importance

The tuned Random Forest model identified the following features as the strongest predictors of customer churn:

| Rank | Feature |
|------|----------|
| 1 | Contract |
| 2 | Monthly Charges |
| 3 | Total Charges |
| 4 | Tenure |
| 5 | Online Security |
| 6 | Tech Support |
| 7 | Online Backup |
| 8 | Payment Method |
| 9 | Dependents |
| 10 | Internet Service |

---

## 📈 Visualizations

### Customer Churn Distribution

![Customer Churn Distribution](images/churn_distribution.png)

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### Feature Importance

![Feature Importance](images/feature_importance.png)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Pickle

---

## 📁 Repository Structure

```text
Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction_using_ML.ipynb
├── README.md
├── requirements.txt
├── customer_churn_model.pkl
├── encoders.pkl
├── images/
│   ├── churn_distribution.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── dataset/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## 🚀 How to Run

```bash
git clone https://github.com/haritharajan04/Customer_Churn_Prediction_using_ML.ipynb.git

cd Customer_Churn_Prediction_using_ML.ipynb

pip install -r requirements.txt

jupyter notebook
```

Open:

```
Customer_Churn_Prediction_using_ML.ipynb
```

---

## 💡 Key Insights

- Random Forest achieved the highest cross-validation accuracy (84%) among all evaluated models.
- Contract type was the most influential feature affecting customer churn.
- Monthly Charges, Total Charges, and Tenure also played significant roles in churn prediction.
- Applying SMOTE improved the model's ability to identify churn cases in the imbalanced dataset.

---

## 🔮 Future Improvements

- Perform advanced hyperparameter optimization using Optuna or Bayesian Optimization.
- Develop a Streamlit web application for real-time churn prediction.
- Explore ensemble and deep learning approaches for improved predictive performance.
- Deploy the trained model as a REST API for production use.

---

## 👩‍💻 Author

**Haritha Rajan**

GitHub: https://github.com/haritharajan04
