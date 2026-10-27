
from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load dataset
dataset = fetch_ucirepo(id=320)

X = dataset.data.features.copy()
y = dataset.data.targets["G3"]

# 2. Remove previous grades
X = X.drop(columns=["G1", "G2"], errors="ignore")

# 3. Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Compare trees of different depths
depths = [1, 2, 3, 4, 5, 8, 12, None]

train_errors = []
test_errors = []

for depth in depths:
    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_mae = mean_absolute_error(
        y_train, train_predictions
    )
    test_mae = mean_absolute_error(
        y_test, test_predictions
    )

    train_errors.append(train_mae)
    test_errors.append(test_mae)

    print(
        f"Depth: {str(depth):>4} | "
        f"Train MAE: {train_mae:.2f} | "
        f"Test MAE: {test_mae:.2f}"
    )

# 6. Plot the results
labels = [str(d) for d in depths]

plt.figure(figsize=(9, 5))
plt.plot(labels, train_errors, marker="o", label="Training MAE")
plt.plot(labels, test_errors, marker="o", label="Test MAE")

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Mean Absolute Error (lower is better)")
plt.title("Underfitting and Overfitting in Decision Trees")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
