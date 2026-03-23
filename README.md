# Customer Segmentation using MLflow

## 👩‍💻 Student Details
- Name: Murugavel Rajan S
- Roll Number: 727823TUAM028  
- Department: BE - CSE (AIML)  

---

## 📌 Project Overview
This project focuses on **Customer Segmentation** using machine learning and MLflow.

The goal is to group customers into different segments based on their behavior and financial attributes. This helps businesses understand customer patterns and improve targeted marketing strategies.

This is an **unsupervised learning (clustering)** problem.

---

## 📊 Dataset Description
A synthetic dataset is generated using `make_blobs()` to simulate real-world customer data.

### Features:
- Age  
- Income  
- Spending Score  
- Savings  
- Purchase Frequency  
- Loan Amount  

There is no target variable, as the goal is to group customers based on similarity.

---

## ⚙️ Technologies Used
- Python  
- Scikit-learn  
- MLflow  
- Pandas  
- NumPy  

---

## 🤖 Models Used
- K-Means Clustering  
- Agglomerative Clustering  

---

## 🔁 MLflow Experiment Tracking
- Experiment Name: `SKCT_727823TUAM028_CustomerSegmentation`
- Total Runs: 12+

### Metrics Logged:
- Silhouette Score  
- Davies-Bouldin Index  

### Operational Metrics:
- training_time_seconds  
- model_size_mb  
- n_features  
- random_seed  

### Tags:
- student_name  
- roll_number  
- dataset  

---

## 🏆 Best Model
- Model: Agglomerative Clustering  
- Best Silhouette Score: ~0.60  

The best model was saved as: best_model.pkl
