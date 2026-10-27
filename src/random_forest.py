
from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
dataset = fetch_ucirepo(id=320)

X = dataset.data.features.copy()
y = dataset.data.targets["G3"]

# 2. Remove previous grades to avoid using them as features
X = X.drop(columns=["G1", "G2"], errors="ignore")

# 3. Convert categorical features into numerical features
X = pd.get_dummies(X, drop_first=True)

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
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

# Store results for comparison
results = []
rf_predictions = None

# 6. Train and evaluate both models
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

    test_r2 = r2_score(
        y_test, test_predictions
    )

    results.append({
        "Model": name,
        "Training MAE": train_mae,
        "Testing MAE": test_mae,
        "Testing R2": test_r2
    })

    # Save actual Random Forest predictions for the graph
    if name == "Random Forest":
        rf_predictions = test_predictions

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Testing MAE:  {test_mae:.2f}")
    print(f"Testing R²:   {test_r2:.3f}")

# 7. Create a comparison table
results_df = pd.DataFrame(results)

print("\nModel Comparison")
print("================")
print(results_df.round(3).to_string(index=False))

# 8. Plot model comparison
model_names = results_df["Model"].tolist()
test_mae_values = results_df["Testing MAE"].tolist()
test_r2_values = results_df["Testing R2"].tolist()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

axes[0].bar(model_names, test_mae_values)
axes[0].set_title("Test MAE (Lower Is Better)")
axes[0].set_ylabel("Mean Absolute Error")
axes[0].tick_params(axis="x", rotation=15)

axes[1].bar(model_names, test_r2_values)
axes[1].axhline(y=0, linestyle="--")
axes[1].set_title("Test R² (Higher Is Better)")
axes[1].set_ylabel("R² Score")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()

# Save the comparison graph
plt.savefig(
    "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 9. Plot actual vs predicted grades for Random Forest
plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.7
)

# Reference line for perfect predictions
minimum = min(y_test.min(), rf_predictions.min())
maximum = max(y_test.max(), rf_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Final Grade")
plt.ylabel("Predicted Final Grade")
plt.title("Actual vs Predicted Student Grades")
plt.legend()
plt.tight_layout()

# Save the prediction graph
plt.savefig(
    "student_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraphs saved successfully:")
print("1. model_comparison.png")
print("2. student_predictions.png")
