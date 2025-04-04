import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import pickle

# Load data
X_train, y_train = joblib.load("train_data.pkl")
X_test, y_test = joblib.load("test_data.pkl")

y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train_oh,
                    validation_split=0.2,
                    epochs=20,
                    batch_size=32)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test_oh)
y_pred = np.argmax(model.predict(X_test), axis=1)
report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])

print(f"\nTest Accuracy: {test_acc:.3f}")
print("\nDeep Learning Classification Report:\n", report)

# Save training history
with open("dl_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
