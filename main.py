import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

# Step 1: Load the Excel file
df = pd.read_excel("test_orange.xlsx")

# Step 2: Split target (y)
y = df["Scaling"]

# Step 3: Drop target and select only numeric features
X = df.drop(columns=["Scaling"])
X_numeric = X.select_dtypes(include='number')  # Remove date/time and text columns

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

# Step 6: Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Print model equation
coefficients = model.coef_[0]
intercept = model.intercept_[0]
feature_names = X_numeric.columns

print("\nLogistic Regression Equation:")
equation = "logit(p) = {:.3f}".format(intercept)
for coef, fname in zip(coefficients, feature_names):
    equation += " + ({:.3f} * {})".format(coef, fname)
print(equation)
print("\nProbability(p) = 1 / (1 + exp(-logit(p)))")

# Step 8: Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

# Step 9: McFadden's pseudo R²
ll_full = -log_loss(y_test, model.predict_proba(X_test), normalize=False)
ll_null = -log_loss(y_test, [np.mean(y_test)] * len(y_test), normalize=False)
mcfadden_r2 = 1 - (ll_full / ll_null)
print("McFadden's pseudo R²: {:.4f}".format(mcfadden_r2))