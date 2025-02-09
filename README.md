# Airfare Price Predictor

## Overview
The **Airfare Price Predictor** is a machine learning-based web application that predicts flight ticket prices based on various features such as airline, source, destination, departure time, arrival time, and more. The model is deployed using **Streamlit** and is accessible online.

## Live Demo
[Airfare Price Predictor Web App](https://airfare-price-predictor.streamlit.app/)

## Repository Structure
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

## Dataset
The dataset used for this project contains flight details such as:
- Airline
- Date of Journey
- Source and Destination
- Departure & Arrival Time
- Duration
- Total Stops
- Ticket Price (Target Variable)

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/suhaneec/Airfare_Price.Predictor.git
   cd Airfare_Price.Predictor
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Model Training
- Data preprocessing was performed using Pandas and Scikit-learn.
- Feature engineering included encoding categorical variables and scaling numerical features.
- Machine Learning models such as **Random Forest, XGBoost, and Linear Regression** were trained.
- The best-performing model was selected based on RMSE and R² scores.

## Technologies Used
- **Python** (Pandas, NumPy, Scikit-learn, XGBoost)
- **Streamlit** (For web app deployment)
- **Matplotlib & Seaborn** (For visualization)
- **Flask** (Optional API integration)

## Future Enhancements
- Adding more real-time data for better accuracy.
- Implementing deep learning models for further improvements.
- Deploying the model using cloud services.

## Contributors
- **Suhani** - EDA, Designing and Deployment, Model Training
- **Ganagavaram** - Model Training
- **Kedar** - Model Training
- **Nissi** - EDA

## License
This project is open-source under the MIT License.

---
For any issues, feel free to raise a GitHub issue or contact me at suhanichauhan58@gmail.com

