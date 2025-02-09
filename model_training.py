# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 📌 Step 1: Load Dataset with Error Handling
try:
    # Load dataset, ensuring proper handling of corrupt rows
    df = pd.read_csv("/content/updated_flight_dataset.csv", on_bad_lines="skip")  # Skips problematic rows
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit()

# 📌 Step 2: Drop unnecessary columns
df.drop(columns=["Flight Number"], errors="ignore", inplace=True)

# 📌 Step 3: Convert 'Flight Date' to datetime format
df["Flight Date"] = pd.to_datetime(df["Flight Date"], errors="coerce")

# 📌 Step 4: Convert 'Departure Time' & 'Arrival Time' into hours (numerical format)
df["Departure Hour"] = pd.to_datetime(df["Departure Time"], format="%H:%M", errors="coerce").dt.hour
df["Arrival Hour"] = pd.to_datetime(df["Arrival Time"], format="%H:%M", errors="coerce").dt.hour

# Drop original time columns as they are now encoded
df.drop(columns=["Departure Time", "Arrival Time"], errors="ignore", inplace=True)

# 📌 Step 5: One-Hot Encode Categorical Features
categorical_cols = ["Airline", "Class", "Origin", "Destination"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 📌 Step 6: Convert all numerical data to float32 for memory efficiency
for col in df.columns:
    if col != "Flight Date":
        df[col] = df[col].astype("float32")

# 📌 Step 7: Define Features (X) and Target (y)
X = df.drop(columns=["Flight Date", "Price (₹)"])
y = df["Price (₹)"]

# 📌 Step 8: Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 9: Train the Optimized Random Forest Model
rf_model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 📌 Step 10: Evaluate Model Performance
y_pred = rf_model.predict(X_test)

performance_metrics = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,  # Manually compute RMSE
    "R² Score": r2_score(y_test, y_pred)
}

# Display model performance
print("✅ Model Training Completed! Here are the performance metrics:")
print(pd.DataFrame([performance_metrics]))

# 📌 Step 11: Save the Trained Model and Feature Names
joblib.dump(rf_model, "flight_price_rf_model.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("📂 Model and feature columns saved successfully!")
print("👉 Model Path: flight_price_rf_model.pkl")
print("👉 Feature Columns Path: feature_columns.pkl")
