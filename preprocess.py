import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load and clean the raw CSV files
# Extract features and labels from dataset
# Map 'normal' to 0, and all attack types to 1
# Create a column transformer that:
#   - One-hot encodes categorical features
#   - Standard scales numeric features
# Apply transformations and save preprocessed data

# Load data
train_df = pd.read_csv("train_kdd_small.csv", header=None).iloc[1:].reset_index(drop=True)
test_df = pd.read_csv("test_kdd_small.csv", header=None).iloc[1:].reset_index(drop=True)

# Separate features and labels
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1].apply(lambda x: 0 if x == 'normal' else 1).astype(int)

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1].apply(lambda x: 0 if x == 'normal' else 1).astype(int)

# Identify categorical features
categorical_features = [1, 2, 3]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), [i for i in range(X_train.shape[1]) if i not in categorical_features])
    ])

# Fit and transform
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Save preprocessed data
joblib.dump((X_train_preprocessed, y_train), "train_data.pkl")
joblib.dump((X_test_preprocessed, y_test), "test_data.pkl")
