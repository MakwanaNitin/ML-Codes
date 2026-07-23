import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Load Dataset
data = pd.read_csv("D:/4050/DataSets/linearregressiondataset.csv")

# 2. Extract Features and Target
x = data.iloc[:, 0:1]  # Independent feature (DataFrame)
y = data.iloc[:, 1]    # Target variable (Series)

print("Features (X):")
print(x)
print("\nTarget (y):")
print(y)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 4. Create and Fit the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Model Parameters
print('\nCoefficients:', model.coef_)
print('Intercept:', model.intercept_)

# 6. Predict for a Specific Value (FIXED: matching feature name to prevent sklearn warning)
single_input = pd.DataFrame([[20.27]], columns=x.columns)
predicted_profit = model.predict(single_input)
print('\nProfit for the population of 20.27 lakh is:', predicted_profit[0])

# 7. Predict on Test Data
y_pred = model.predict(X_test)
print("Predicted values for test set:\n", y_pred)

# 8. Model Evaluation Metrics
print('\nMean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))