
from ucimlrepo import fetch_ucirepo
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold
)
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

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create model
model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

# 6. Set up 5-fold cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 7. Evaluate using negative MAE
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_mean_absolute_error"
)

# Convert negative MAE to positive MAE
mae_scores = -scores

print("5-Fold Cross-Validation Results")
print("--------------------------------")

for i, score in enumerate(mae_scores, start=1):
    print(f"Fold {i} MAE: {score:.2f}")

print(f"\nMean CV MAE: {mae_scores.mean():.2f}")
print(f"Standard Deviation: {mae_scores.std():.2f}")

# 8. Train on all training data and evaluate the held-out test set
model.fit(X_train, y_train)
predictions = model.predict(X_test)

test_mae = mean_absolute_error(y_test, predictions)

print(f"\nHeld-Out Test MAE: {test_mae:.2f}")
