
from ucimlrepo import fetch_ucirepo
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
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

# 5. Create both models
models = {
    "Decision Tree": DecisionTreeRegressor(
        max_depth=4,
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        random_state=42
    )
}

# 6. Train and compare
for name, model in models.items():
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_mae = mean_absolute_error(
        y_train, train_predictions
    )
    test_mae = mean_absolute_error(
        y_test, test_predictions
    )
    test_r2 = r2_score(y_test, test_predictions)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Testing MAE:  {test_mae:.2f}")
    print(f"Testing R²:   {test_r2:.3f}")

import matplotlib.pyplot as plt

models = ["Decision Tree", "Random Forest"]
test_mae = [2.30, 2.11]
test_r2 = [-0.033, 0.181]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(models, test_mae)
axes[0].set_title("Test MAE (Lower is Better)")
axes[0].set_ylabel("MAE")
axes[0].tick_params(axis="x", rotation=15)

axes[1].bar(models, test_r2)
axes[1].axhline(0, linewidth=1)
axes[1].set_title("Test R² (Higher is Better)")
axes[1].set_ylabel("R²")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.show()
