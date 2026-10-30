from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

dataset = fetch_ucirepo(id=320)
X = dataset.data.features.copy()
y = dataset.data.targets["G3"]

X = X.drop(columns=["G1", "G2"], errors="ignore")

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_mean_absolute_error"
)

mae_scores = -scores

print("5-Fold Cross-Validation Results")
print("--------------------------------")

for i, score in enumerate(mae_scores, start=1):
    print(f"Fold {i} MAE: {score:.2f}")

print(f"\nMean CV MAE: {mae_scores.mean():.2f}")
print(f"Standard Deviation: {mae_scores.std():.2f}")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

test_mae = mean_absolute_error(y_test, predictions)

print(f"\nHeld-Out Test MAE: {test_mae:.2f}")
