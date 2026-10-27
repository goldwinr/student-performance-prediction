# Student Performance Prediction Using Machine Learning

## Project Overview

This project predicts students' final mathematics grades using machine learning. It applies concepts from Kaggle's Introduction to Machine Learning course, including data exploration, model training, model validation, underfitting and overfitting, and Random Forests.

The project compares a Decision Tree Regressor with a Random Forest Regressor to investigate their performance on student data.

## Problem Statement

Predicting student academic performance can help us explore the relationships between student characteristics and final grades. This project uses historical student data to build regression models that predict final mathematics grades based on selected student attributes.

Previous-period grades are excluded from the input features to make the prediction task more challenging and avoid relying directly on earlier grades.

This is an educational machine learning project, not a validated student assessment system.

## Dataset

- **Source:** UCI Machine Learning Repository
- **Dataset:** Student Performance
- **Subject:** Mathematics
- **Target variable:** `G3` (final grade, on a 0–20 scale)
- **Input features:** Student characteristics and other available attributes, excluding `G1` and `G2`

Dataset: https://archive.ics.uci.edu/dataset/320/student+performance

## Machine Learning Workflow

1. **Data Exploration:** Inspected the dataset, column names, missing values, descriptive statistics, and grade distributions.
2. **Feature Selection:** Selected input features and excluded previous-period grades `G1` and `G2`.
3. **Preprocessing:** Converted categorical features into numerical columns using Pandas.
4. **Model Training:** Trained a Decision Tree Regressor.
5. **Model Validation:** Applied five-fold cross-validation using Mean Absolute Error.
6. **Underfitting and Overfitting:** Compared Decision Trees with different maximum depths.
7. **Random Forests:** Trained a Random Forest Regressor and compared its performance with the Decision Tree.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- ucimlrepo

## Model Results

The following results were obtained from the selected 80/20 train-test split.

| Metric | Decision Tree | Random Forest |
|---|---:|---:|
| Training MAE | 1.86 | 1.75 |
| Testing MAE | 2.30 | 2.11 |
| Testing R² | -0.033 | 0.181 |

### Interpretation

The Random Forest achieved a lower test MAE than the Decision Tree on this split. Its R² score was also higher, indicating better performance relative to a mean-prediction baseline.

However, the Random Forest's R² score remains modest. These results do not establish that the model will generalize equally well to other schools or student populations. Additional validation is required before drawing broader conclusions.

## Project Structure

```text
student-performance-prediction/
├── data/
├── notebooks/
├── src/
│   ├── explore_data.py
│   ├── train_model.py
│   ├── validate_model.py
│   ├── compare_trees.py
│   └── random_forest.py
├── README.md
└── .gitignore
```

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd student-performance-prediction
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install pandas matplotlib seaborn scikit-learn ucimlrepo
```

### 4. Run the scripts

```bash
python src/explore_data.py
python src/train_model.py
python src/validate_model.py
python src/compare_trees.py
python src/random_forest.py
```

An internet connection is required to retrieve the dataset through `ucimlrepo`.

## Key Learnings

- Exploring and understanding a real dataset.
- Separating input features from the target variable.
- Training regression models with scikit-learn.
- Evaluating models using MAE and R².
- Understanding cross-validation.
- Identifying possible underfitting and overfitting.
- Comparing a single Decision Tree with a Random Forest ensemble.

## Limitations and Future Improvements

- Compare models using cross-validation before selecting the final model.
- Evaluate performance on additional data splits or independent datasets.
- Investigate feature importance carefully.
- Compare predictions against actual grades using additional visualizations.
- Explore hyperparameter tuning without using the test set for model selection.

## Disclaimer

This project is intended for educational purposes. Its predictions should not be used to make decisions about real students' academic opportunities or abilities.

## Author

Developed as a beginner machine learning project to apply the concepts learned through Kaggle's Introduction to Machine Learning course.
