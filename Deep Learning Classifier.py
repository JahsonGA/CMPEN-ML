import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# One-hot encode labels
y_train_oh = to_categorical(y_train_bin)
y_test_oh = to_categorical(y_test_bin)

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_preprocessed.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # Binary output
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_preprocessed, y_train_oh,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test_preprocessed, y_test_oh)
print(f"\nTest Accuracy: {test_accuracy:.3f}")

# Predict and report
y_pred_prob = model.predict(X_test_preprocessed)
y_pred = np.argmax(y_pred_prob, axis=1)
report = classification_report(y_test_bin, y_pred, target_names=["Normal", "Attack"])
print(report)
