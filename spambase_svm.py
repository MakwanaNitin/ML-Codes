# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (handles the text header automatically)
df = pd.read_csv("D:/4050/DataSets/spambase.csv")

# Dynamic features and target split:
# X = all columns except the last one
# y = the very last column (the spam class label)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data (SVMs are highly sensitive to unscaled features!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SVM model (using RBF kernel by default)
model = SVC(kernel="rbf", random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Display results
print("SVM Accuracy of Spambase:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))