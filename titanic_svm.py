# Import required libraries
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

# Scale data (Mandatory for SVM! Large-scale features will otherwise dominate the margin)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SVM model
# 'rbf' (Radial Basis Function) is the default kernel and handles non-linear boundaries well
model = SVC(kernel="rbf", random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Display results
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))