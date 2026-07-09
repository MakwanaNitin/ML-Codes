# Import required libraries
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 

# Load dataset
df = pd.read_csv("D:/4050/DataSets/titanic.csv")

# --- DATA PREPROCESSING ---
# 1. Drop identifier columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

# 2. Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 3. Convert categorical text features to numbers (One-Hot Encoding)
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
# --------------------------

# Features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# NOTE: Decision Trees are scale-invariant, meaning they do NOT require 
# StandardScaler like KNN does. We can pass the raw processed data directly!

# Create Decision Tree model
# Setting max_depth helps prevent the tree from overfitting the training data
model = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Display results
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))