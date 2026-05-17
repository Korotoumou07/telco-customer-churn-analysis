# Telco Customer Churn Prediction using Machine Learning

## Project Overview
This project focuses on predicting customer churn in a telecommunications company using Machine Learning techniques.

The objective is to identify customers who are likely to leave the company and help support customer retention strategies through predictive analytics.

The project combines:
- Data preprocessing
- Feature engineering
- Machine Learning modeling
- Model evaluation
- Interactive dashboarding
- Web application deployment

---

## Business Problem
Customer churn represents a major challenge for telecom companies because losing customers directly impacts revenue.

This project aims to:
- Detect high-risk customers
- Understand churn behavior
- Support decision-making with predictive insights
- Improve retention strategies

---

## Dataset
Dataset: Telco Customer Churn Dataset

Main information:
- Customer demographics
- Contract information
- Internet services
- Billing information
- Customer tenure
- Churn status

Target variable:
- Churn (Yes / No)

---

## Technologies Used

### Programming & Analysis
- Python
- Jupyter Notebook

### Libraries
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib
- Seaborn

### Visualization & Deployment
- Power BI
- Streamlit

---

# Workflow

## 1. Exploratory Data Analysis (EDA)
Performed exploratory analysis to understand:
- Customer demographics
- Churn distribution
- Contract impact
- Payment behavior
- Service usage patterns

### 📸 Screenshots 
- Churn Count Distribution
<img width="827" height="374" alt="image" src="https://github.com/user-attachments/assets/c17514bd-da28-44c0-a955-71432a43f276" />
- Tenure vs Churn visualization
<img width="1184" height="361" alt="image" src="https://github.com/user-attachments/assets/aa9a4aee-aff7-4a94-8c69-58f8ccab4701" />
- Correlation heatmap
<img width="990" height="366" alt="image" src="https://github.com/user-attachments/assets/42fd178f-39bf-4d4d-b7cd-b16f8ff57712" />


---

## 2. Feature Engineering
Performed:
- Missing value handling
- Label Encoding
- One-Hot Encoding
- Feature scaling
- SMOTE balancing for class imbalance

### 📸 Screenshots 
- Encoding Variables and Train/Test Split
  <img width="1192" height="634" alt="image" src="https://github.com/user-attachments/assets/c187fff9-17ee-41f5-8ccc-e8f43a9840d1" />

- SMOTE balancing visualization
  <img width="1177" height="413" alt="image" src="https://github.com/user-attachments/assets/fed546b1-28d7-4657-939d-e7293b02144a" />

- Encoded dataset preview
<img width="1191" height="264" alt="image" src="https://github.com/user-attachments/assets/20d5101d-3582-45e0-b6b0-68122ceb8ffd" />

---

## 3. Machine Learning Modeling
Models trained and compared:
- Logistic Regression
- Random Forest
- XGBoost
- Voting Classifier

Evaluation metrics:
- Accuracy
- Recall
- Precision
- F1-Score
- ROC-AUC

### 📸 Screenshots 
- Model comparison table
<img width="531" height="187" alt="image" src="https://github.com/user-attachments/assets/7e200892-cfd8-4ce8-9a1c-2d32283a2681" />

- ROC Curve comparison
  <img width="715" height="502" alt="image" src="https://github.com/user-attachments/assets/d818115e-c5e1-4efb-bfa3-e763ade32d50" />
- Final Model - Classification Report
<img width="423" height="217" alt="image" src="https://github.com/user-attachments/assets/f83180a1-b6e7-406f-bd1d-73e23a6b74f8" />


---

## 4. Power BI Dashboard
Developed an interactive dashboard for churn monitoring.

Dashboard sections:
- Customer Overview
- Churn Analysis
- Contract & Billing
- Customer Risk Analysis
- Model Performance

### 📸 Screenshots 
- Full dashboard overview
<img width="1207" height="671" alt="image" src="https://github.com/user-attachments/assets/b4d144c9-0b20-4188-975a-fccce2f35813" />


---

## 5. Streamlit Web Application
Built a Streamlit web application for real-time churn prediction.

Main functionalities:
- User input form
- Real-time prediction
- Prediction probability display
- Customer risk visualization

### 📸 Screenshots  Streamlit interface
<img width="1901" height="851" alt="image" src="https://github.com/user-attachments/assets/5a9b886a-fc70-42f4-8f45-ec6b41a05e45" />
<img width="1871" height="832" alt="image" src="https://github.com/user-attachments/assets/51695c70-a17e-4d79-89b9-a00229ddf062" />


---

## Results
Key achievements:
- Successfully identified high-risk customers
- Compared multiple Machine Learning models
- Built a business-oriented dashboard
- Deployed a predictive web application

---

## Installation

### Clone the repository
```bash
git clone https://github.com/Korotoumou07/telco-customer-churn-analysis.git
````

### Navigate into the project folder

```bash
cd project_folder
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## Future Improvements

Possible improvements:

* Hyperparameter optimization
* Deep Learning integration
* Cloud deployment
* Real-time API integration
* Advanced customer segmentation

