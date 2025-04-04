from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Dictionary to hold classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier()
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train_preprocessed, y_train_bin)
    y_pred = model.predict(X_test_preprocessed)
    report = classification_report(y_test_bin, y_pred, output_dict=True)
    results[name] = report

# Display results
import pandas as pd
summary_df = pd.DataFrame({model: {
    "Accuracy": round(metrics["accuracy"], 3),
    "Precision (Attack)": round(metrics["1"]["precision"], 3),
    "Recall (Attack)": round(metrics["1"]["recall"], 3),
    "F1 Score (Attack)": round(metrics["1"]["f1-score"], 3),
} for model, metrics in results.items()}).T

import ace_tools as tools; tools.display_dataframe_to_user(name="Model Evaluation Summary", dataframe=summary_df)
