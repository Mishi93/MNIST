import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load CSV
data = pd.read_csv("mnist.csv")  # label + 784 pixels
X = data.drop("label", axis=1)
y = data["label"]

# Normalize pixels to [0,1]
X = X / 255.0

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression (multinomial)
model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,  # increase iterations to converge
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "mnist_model.pkl")