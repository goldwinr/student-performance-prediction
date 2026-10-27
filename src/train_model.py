
from ucimlrepo import fetch_ucirepo
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
dataset = fetch_ucirepo(id=320)

X = dataset.data.features.copy()
y = dataset.data.targets["G3"]

# 2. Remove previous grades to avoid relying on them
X = X.drop(columns=["G1", "G2"], errors="ignore")

# 3. Convert categorical features into numerical features
X = pd.get_dummies(X, drop_first=True)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create the model
model = DecisionTreeRegressor(max_depth=4, random_state=42)

# 6. Train the model
model.fit(X_train, y_train)

# 7. Make predictions
predictions = model.predict(X_test)

# 8. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Decision Tree Model Results")
print("----------------------------")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# 9. Compare actual and predicted grades
results = pd.DataFrame({
    "Actual Grade": y_test,
    "Predicted Grade": predictions.round(2)
})

print("\nSample Predictions:")
print(results.head(10))
