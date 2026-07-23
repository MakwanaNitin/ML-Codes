# KNN Regression using CSV File

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Load CSV Dataset
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

# Display Results
print("Actual\tPredicted")
for a, p in zip(y_test, y_pred):
    print(a, "\t", round(p, 2))

# Predict New House Price
new_house = [[2500, 4]]
prediction = model.predict(new_house)

print("\nPredicted Price:", round(prediction[0], 2))