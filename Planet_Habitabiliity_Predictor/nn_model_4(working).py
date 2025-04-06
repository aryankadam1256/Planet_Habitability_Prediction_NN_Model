import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1) Read the dataset
data = pd.read_csv("dataset_3.csv")

# 2) Select features (X) and target (Y)
X = data[['Orbital_Distance_AU', 'Planet_Radius_Earth', 'Surface_Temperature_K']]
Y = data['Habitable']

# 3) Encode the target variable
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)  # Ensures binary labels (0 or 1)

# 4) Split the data into training (80%) and testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 5) Normalize (scale) the features using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6) Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# 7) Build an improved neural network model
model = Sequential([
    Dense(8, activation="relu", input_shape=(3,)),  # Increased neurons for better learning
     Dense(3, activation="relu"),  # Additional hidden layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# 8) Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 9) Print model summary
model.summary()

# 10) Train the model with validation & class weights
history=model.fit(X_train, Y_train, epochs=150, batch_size=10, validation_data=(X_test, Y_test), class_weight=class_weight_dict)

# 11) Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
print("The accuracy of the model is:", accuracy * 100, "%")

# 12) Predict on custom new planet data
new_planets = pd.DataFrame(
    [[1.0, 1.0, 288],  # Earth (expected: habitable)
     [1.52, 0.53, 210],  # Mars (expected: non-habitable)
     [0.39, 0.38, 700],  # Mercury (expected: non-habitable)
     [0.72, 0.95, 737]],  # Venus (expected: non-habitable)
    columns=X.columns  # Maintain feature names to avoid warnings
)

# 13) Scale custom input using trained scaler
new_planets_scaled = scaler.transform(new_planets)

# 14) Predict using the trained model
predictions = model.predict(new_planets_scaled)

# 15) Print predictions
for i, pred in enumerate(predictions):
    status = "Habitable" if pred >= 0.5 else "Not Habitable"
    print(f"Planet {i+1}: {status} (Probability: {pred[0]:.2f})")

 
#9) Plot training & validation loss/accuracy
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='red')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()