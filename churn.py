## Full Working Churn Prediction Code

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1️⃣ Create sample dataset (replace with your CSV later)
data = {
    "tenure": [1, 2, 3, 10, 15, 20, 2, 5, 7, 30],
    "monthly_charges": [70, 80, 65, 90, 100, 110, 60, 75, 85, 120],
    "churn": [1, 1, 1, 0, 0, 0, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

# 2️⃣ Features and target
X = df[["tenure", "monthly_charges"]]
y = df["churn"]

# 3️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5️⃣ Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6️⃣ Predict
y_pred = model.predict(X_test)

# 7️⃣ Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

# 8️⃣ Predict new customer
new_customer = [[5, 80]]  # tenure=5 months, $80/month
new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled)

print("Churn prediction:", prediction[0])  # 1=leave, 0=stay