import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "ADSI_Table.csv"
df = pd.read_csv(file_path)

# Drop irrelevant columns
df = df.drop(columns=["Sl. No.", "State/UT/City"])

# Check for missing values
if df.isnull().sum().any():
    df = df.dropna()

# Define features and target
X = df.drop(columns=["Total Traffic Accidents - Died"])
y = df["Total Traffic Accidents - Died"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save the model and scaler
joblib.dump(model, "traffic_accident_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Plot predictions
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Deaths")
plt.ylabel("Predicted Deaths")
plt.title("Actual vs Predicted Traffic Accident Deaths")
plt.grid(True)
plt.show()
