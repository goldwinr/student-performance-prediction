from ucimlrepo import fetch_ucirepo
import pandas as pd

# Load the Student Performance dataset
dataset = fetch_ucirepo(id=320)

# Select the mathematics dataset
X = dataset.data.features
y = dataset.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

print("Dataset loaded successfully!")
print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFinal grade distribution:")
print(df["G3"].value_counts().sort_index())

# 1. Dataset information
print("\n--- Dataset Information ---")
print(df.info())

# 2. Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())

# 3. Check duplicate rows
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# 4. Explore final grades
print("\n--- Final Grade Statistics ---")
print(df["G3"].describe())

# 5. Visualize final grade distribution
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=20, kde=True)
plt.title("Distribution of Final Student Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# 6. Visualize study time against final grade
plt.figure(figsize=(8, 5))
sns.boxplot(x="studytime", y="G3", data=df)
plt.title("Study Time vs Final Grade")
plt.xlabel("Weekly Study Time Category")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.show()