# KNN Regression with Graph

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Load Dataset
data = pd.read_csv("D:/4050/DataSets/MultivariateLinearRegressionDataset.csv")

# Features and Target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create KNN Model
model = KNeighborsRegressor(n_neighbors=3)

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print Actual and Predicted Values
print("Actual\tPredicted")
for a, p in zip(y_test, y_pred):
    print(a, "\t", round(p, 2))

# -------------------------------
# Graph
# -------------------------------
plt.figure(figsize=(8,5))

plt.plot(range(len(y_test)), y_test.values, marker='o', label="Actual")
plt.plot(range(len(y_pred)), y_pred, marker='x', label="Predicted")

plt.title("Actual vs Predicted House Price")
plt.xlabel("Test Data")
plt.ylabel("House Price")
plt.legend()
plt.grid(True)

plt.show()