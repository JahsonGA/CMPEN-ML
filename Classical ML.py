import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd

# Load data
X_train, y_train = joblib.load("train_data.pkl")
X_test, y_test = joblib.load("test_data.pkl")

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier()
}

# Evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        "Accuracy": round(report["accuracy"], 3),
        "Precision (Attack)": round(report["1"]["precision"], 3),
        "Recall (Attack)": round(report["1"]["recall"], 3),
        "F1 Score (Attack)": round(report["1"]["f1-score"], 3)
    }

# Save and display results
results_df = pd.DataFrame(results).T
print("\nClassical Model Performance:\n")
print(results_df)
