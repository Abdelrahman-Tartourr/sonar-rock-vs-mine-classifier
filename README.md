# Rock vs Mine Prediction

A machine learning classifier that predicts whether a sonar signal bounced off a rock or a metal mine, using 60 sonar frequency readings per sample.

## Problem

Sonar signals reflected off objects on the seafloor can look very similar whether the object is a harmless rock or a metal mine. Telling them apart reliably from raw sonar readings is a classic hard classification problem — this project builds a model to do that automatically instead of relying on manual interpretation.

## Approach

Started with a baseline Logistic Regression model, then compared it against SVM, Random Forest, and KNN using 5-fold stratified cross-validation to pick the best performer with evidence rather than guessing.

- **Why cross-validation over a single train/test split**: with only 208 total samples, one split's accuracy can swing several points just from which rows happen to land in the test set. CV averages over 5 different splits for a more trustworthy estimate.
- **Why SVM (RBF kernel) won**: it captured non-linear relationships between the 60 sonar frequency features that Logistic Regression's linear boundary missed, and had the most consistent (lowest variance) scores across folds.
- **Tricky part**: feature scaling. Logistic Regression and SVM are both sensitive to feature scale, so each model was wrapped in a `Pipeline` with `StandardScaler` to scale correctly *inside* each cross-validation fold (scaling before splitting would leak test data statistics into training).
- **With more time/data**: I'd want more samples — 208 rows is small for a 60-feature problem, and the train/test accuracy gap (97.9% vs 85.7%) suggests the model is still somewhat overfit to this particular dataset.

## Stack

`Python` `pandas` `scikit-learn` `matplotlib` `seaborn` `Jupyter`

## Result

Final model: **SVM (RBF kernel)**, selected via cross-validation.

| Metric | Score |
|---|---|
| Cross-validation accuracy (mean, 5-fold) | 81.8% |
| Training accuracy | 97.9% |
| Test accuracy | 85.7% |

**Classification report (test set, n=21):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Rock | 0.83 | 0.91 | 0.87 |
| Mine | 0.89 | 0.80 | 0.84 |

Confusion matrix and model comparison boxplot are included in the notebook output.

## What I'd improve with more time

1. **More data** — 208 samples is small for 60 features; performance estimates (especially the 21-sample test set) have meaningful uncertainty (±~5% per sample).
2. **Reduce overfitting** — the train/test accuracy gap (97.9% vs 85.7%) suggests hyperparameter tuning (e.g. `GridSearchCV` on SVM's `C` and `gamma`) would help generalization.
3. **Feature importance** — using permutation importance to see which of the 60 sonar frequencies actually drive the Rock/Mine decision, since SVM doesn't expose feature importances directly like Random Forest does.

## How to run it

```bash
# 1. Clone the repo / download the folder
# 2. Make sure sonar_data.csv is in the same folder as the notebook
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook Rock_vs_Mine_Prediction_v2.ipynb
# Run all cells top to bottom
```

---
*Built on public data (UCI Sonar dataset). No proprietary or employer data used.*
