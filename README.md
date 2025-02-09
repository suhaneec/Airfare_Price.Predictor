# ✈️ Airfare Price Predictor

## 🌟 Overview
The **Airfare Price Predictor** is an AI-powered flight price prediction system that provides real-time airfare insights and trend analysis. The model leverages machine learning techniques to analyze various flight parameters and predict ticket prices with high accuracy.

🔗 **Live Demo:** [Airfare Price Predictor Web App](https://airfare-price-predictor.streamlit.app/)

---

## 📂 Repository Structure
```
📂 Airfare_Price.Predictor
│-- 📂 data              # Dataset used for training
│-- 📂 notebooks         # Jupyter notebooks for EDA and model training
│-- 📂 models            # Saved trained models
│-- 📂 src               # Source code for the Streamlit app
│-- 📜 requirements.txt  # Required Python libraries
│-- 📜 app.py            # Main script for the Streamlit app
│-- 📜 README.md         # Project documentation
```

---

## 📊 Dataset
The dataset contains the following key flight details:
- 🛫 **Airline**
- 📅 **Date of Journey**
- 🏙️ **Source & Destination**
- ⏰ **Departure & Arrival Time**
- ⏳ **Duration**
- 🛑 **Total Stops**
- 💰 **Ticket Price (Target Variable)**

---

## 🛠️ Installation
To run this project locally, follow these steps:

1️⃣ Clone the repository:
   ```bash
   git clone https://github.com/suhaneec/Airfare_Price.Predictor.git
   cd Airfare_Price.Predictor
   ```
2️⃣ Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3️⃣ Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4️⃣ Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 🤖 Model Training
- 🔍 **Exploratory Data Analysis (EDA)** performed using **Pandas & Seaborn**.
- ⚙️ **Feature Engineering** included encoding categorical variables and scaling numerical features.
- 🚀 **Machine Learning Models** trained: **Random Forest, XGBoost, and Linear Regression**.
- 🏆 The best-performing model was selected based on **RMSE and R² scores**.

### 📌 Libraries Used for Model Training
```python
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

---

## 🛠️ Technologies Used
- 🐍 **Python** (Pandas, NumPy, Scikit-learn, XGBoost)
- 🎨 **Streamlit** (For web app deployment)
- 📊 **Matplotlib & Seaborn** (For visualization)
- 🌐 **Flask** (Optional API integration)

---

## 🚀 Future Enhancements
- 📡 Incorporating real-time flight data for better accuracy.
- 🧠 Implementing deep learning models for further improvements.
- ☁️ Deploying the model on **cloud platforms**.

---

## 👨‍💻 Contributors
- **👩‍💻 Suhani** - EDA, Designing, Deployment, Model Training
- **🧑‍💻 Farooq** - Model Training
- **🧑‍💻 Kedar** - Model Training
- **👩‍💻 Nissi** - EDA

---

## 📜 License
This project is open-source under the **MIT License**.

For any issues, feel free to **raise a GitHub issue** or contact the contributors! 🚀

