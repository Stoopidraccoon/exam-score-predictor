import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 1: Create Dataset
data = {
    "Study_Hours": [1.5, 2.0, 3.0, 3.5, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0],
    "Exam_Score": [48, 52, 58, 63, 70, 74, 81, 86, 92, 97]
}
df = pd.DataFrame(data)
print("--- Exam Score Dataset ---")
print(df)

# Step 2: Separate X and y
X = df[["Study_Hours"]]
y = df["Exam_Score"]

# Step 3: Train-Test Split (80% Train / 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel Training Completed!")

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Step 6: Regression Equation & Manual Verification for 6.5 Hours
intercept = model.intercept_
slope = model.coef_[0]

print("\n--- Linear Regression Equation ---")
print(f"Exam Score = {intercept:.2f} + {slope:.2f} * (Study Hours)")

hours = 6.5
manual_pred = intercept + (slope * hours)
print(f"\nManual Calculation for {hours} Study Hours: {manual_pred:.2f}%")

# Step 7: Save Trained Model
joblib.dump(model, "exam_score_model.pkl")
print("\nModel saved successfully as 'exam_score_model.pkl'!")