import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
import pickle

to_categorical = keras.utils.to_categorical
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout

# Load data
X_train, y_train = joblib.load("train_data.pkl")
X_test, y_test = joblib.load("test_data.pkl")

y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),      # First hidden layer
    Dropout(0.3),                                                       # Dropout to prevent overfitting
    Dense(32, activation='relu'),                                       # Second hidden layer
    Dropout(0.3),                                                       # Dropout to prevent overfitting
    Dense(2, activation='softmax')                                      # Output layer for binary classification
])

# Compile the model with optimizer and loss function
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
#   validate on 20% of training data
history = model.fit(X_train, y_train_oh,
                    validation_split=0.2,
                    epochs=20,
                    batch_size=32)

# Evaluate model
#   test data and generate classification report
test_loss, test_acc = model.evaluate(X_test, y_test_oh)
y_pred = np.argmax(model.predict(X_test), axis=1)
report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])

print(f"\nTest Accuracy: {test_acc:.3f}")
print("\nDeep Learning Classification Report:\n", report)

# Save training history
with open("dl_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
