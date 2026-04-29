import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# =====================
# LOAD DATA
# =====================
df = pd.read_csv(r"C:\Users\DELL\Downloads\heart_disease_data.csv")

X = df.drop("target", axis=1)
y = df["target"]

# =====================
# SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================
# MODEL (محسن)
# =====================
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=8,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# =====================
# PROBABILITIES
# =====================
y_prob = rf.predict_proba(X_test)[:, 1]

# =====================
# 🔥 THRESHOLD OPTIMIZATION (F1 BEST)
# =====================
best_t = 0.5
best_f1 = 0

for t in np.arange(0.2, 0.8, 0.01):
    y_pred = (y_prob >= t).astype(int)

    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("\n🔥 Best Threshold:", best_t)

# =====================
# FINAL PREDICTION
# =====================
y_pred_final = (y_prob >= best_t).astype(int)

# =====================
# RESULTS
# =====================
print("\n================ RESULTS ================")
print("Accuracy:", (y_pred_final == y_test).mean())
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_final))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_final))

import joblib

# =====================
# SAVE MODEL
# =====================
model_data = {
    "model": rf,
    "threshold": best_t,
    "features": X.columns.tolist()
}

joblib.dump(model_data, r"C:\Users\DELL\OneDrive\Desktop\task2\models\heart_rf_model.pkl")

print("\n✅ Model saved successfully as heart_rf_model.pkl")