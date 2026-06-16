# Import Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Check Dataset Shape
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Normalize Pixel Values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build ANN Model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_split=0.2
)

# Evaluate Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", test_accuracy)

# Predict Sample Images
predictions = model.predict(X_test)

# Display First 5 Predictions
for i in range(5):
    plt.imshow(X_test[i], cmap='gray')
    plt.title(
        f"Actual: {y_test[i]} | Predicted: {predictions[i].argmax()}"
    )
    plt.show()
