from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Drop header row (first row is the column names)
train_df_clean = train_df.iloc[1:].copy()
test_df_clean = test_df.iloc[1:].copy()

# Reset index just in case
train_df_clean.reset_index(drop=True, inplace=True)
test_df_clean.reset_index(drop=True, inplace=True)

# Separate features and labels
X_train = train_df_clean.iloc[:, :-1]
y_train = train_df_clean.iloc[:, -1]

X_test = test_df_clean.iloc[:, :-1]
y_test = test_df_clean.iloc[:, -1]

# Convert labels to binary: normal -> 0, not_normal -> 1
y_train_bin = y_train.apply(lambda x: 0 if x == 'normal' else 1).astype(int)
y_test_bin = y_test.apply(lambda x: 0 if x == 'normal' else 1).astype(int)

# Identify categorical columns by name (based on column 1, 2, 3)
categorical_features = [1, 2, 3]

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), [i for i in range(X_train.shape[1]) if i not in categorical_features])
    ]
)

# Fit on training, transform both train and test
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Final shapes and sample
X_train_preprocessed.shape, X_test_preprocessed.shape, y_train_bin.value_counts(), y_test_bin.value_counts()
