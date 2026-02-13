# 🏦 Bank Customer Churn Prediction (Deep Learning - ANN)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-85.65%25-green.svg)

## 📖 Project Overview
This project leverages Artificial Neural Networks (ANN) to predict the likelihood of bank customer churn. By processing financial and demographic data, the model provides actionable insights to help financial institutions reduce attrition through data-driven decisions.

## 🛠️ Technical Workflow & Methodology

### 1. Data Cleaning & Feature Engineering
The dataset was pre-processed by removing non-predictive identifiers (RowNumber, CustomerID, Surname). Categorical variables were transformed using numeric mapping for consistency:
- **Gender:** Female (0), Male (1)
- **Geography:** France (0), Spain (1), Germany (2)

### 2. Feature Scaling (Standardization)
To optimize the neural network's learning process, all numerical features were scaled using **Standardization**. This ensures that variables like `Balance` and `Credit Score` contribute equally to the model's weight updates, preventing bias toward features with larger numerical ranges.

### 3. Artificial Neural Network Architecture
The predictive engine is a Sequential ANN featuring:
- **Hidden Layers:** Two layers with 6 neurons each, utilizing **ReLU** activation to learn complex non-linear patterns.
- **Output Layer:** A single neuron using **Sigmoid** activation to output the probability of churn (Binary Classification).
- **Optimization:** Trained using the **Adam** optimizer and **Binary Cross-entropy** loss function for 200 epochs.

## 📊 Performance Metrics & Results

The following tables summarize the performance of the ANN model across the Training and Testing datasets:

### Table 1: Model Accuracy Summary
| Dataset | Accuracy Rate | Samples Evaluated | Correct Predictions |
| :--- | :---: | :---: | :---: |
| **Training Set** | **86.13%** | 8,000 | 6,890 |
| **Test Set** | **85.65%** | 2,000 | 1,713 |

### Table 2: Detailed Confusion Matrix (Test Set)
| Category | Predicted: Stay (0) | Predicted: Churn (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual: Stay (0)** | **1,532 (True Negative)** | 75 (False Positive) | 1,607 |
| **Actual: Churn (1)** | 212 (False Negative) | **181 (True Positive)** | 393 |

## 📈 Key Technical Insights
- **High Generalization:** The model shows exceptional stability, with a minimal variance (0.48%) between training and testing accuracy, indicating no overfitting.
- **Optimized Learning:** The balance between batch size (50) and epochs (200) allowed the model to reach a stable convergence plateau efficiently.
- **Predictive Power:** EDA confirmed that `Age`, `Balance`, and `Number of Products` are the primary drivers influencing customer loyalty.

## 🛠️ Tech Stack
- **Deep Learning Framework:** TensorFlow, Keras
- **Data Visualization:** Plotly Express, Matplotlib
- **Preprocessing & Metrics:** Scikit-Learn
- **Data Handling:** Pandas, NumPy
