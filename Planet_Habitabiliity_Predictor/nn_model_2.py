import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Create a Dataset (features: temperature, gravity, atmospheric pressure, water availability, atmospheric composition)
X = np.array([
    [15, 9.8, 1.0, 1, 1],   # Earth (Habitable)
    [-55, 3.7, 0.006, 0, 0], # Mars (Not Habitable)
    [464, 8.9, 92, 0, 0],    # Venus (Not Habitable)
    [-179, 1.35, 1.5, 0, 0], # Titan (Not Habitable)
    [-50, 24.8, 1.3, 0, 1]   # Europa (Potentially Habitable)
])

# Labels (0 = Not Habitable, 1 = Habitable)
y = np.array([1, 0, 0, 0, 1])  # Earth, Europa = 1 (habitable), others = 0

# Step 2: Build the Model
model = Sequential()

# Input Layer (5 features)
model.add(Dense(units=8, input_dim=5, activation='relu'))  # 8 neurons in the first hidden layer

# Hidden Layer
model.add(Dense(units=4, activation='relu'))  # 4 neurons in the second hidden layer

# Output Layer (1 output, because it's binary classification)
model.add(Dense(units=1, activation='sigmoid'))  # Sigmoid for binary output (0 or 1)

# Step 3: Compile the Model
model.compile(loss='binary_crossentropy',  # Loss function for binary classification
              optimizer='adam',            # Adam optimizer (works well for most problems)
              metrics=['accuracy'])       # We want to track accuracy during training

# Step 4: Train the Model
model.fit(X, y, epochs=50, batch_size=1)  # 50 epochs, batch_size=1 for simplicity

# Step 5: Make Predictions on New Data
new_planets = np.array([
    [20, 9.8, 1.0, 1, 1],  # Earth-like
    [-80, 1.5, 0.5, 0, 0]   # Hypothetical Planet
])

# Predict if the planets are habitable or not
predictions = model.predict(new_planets)

# Output predictions (0 = Not Habitable, 1 = Habitable)
for i, pred in enumerate(predictions):
    result = "Habitable" if pred >= 0.5 else "Not Habitable"
    print(f"Planet {i+1}: {result} (Predicted Probability: {pred[0]:.4f})")
