import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random

# Minimal seeding for consistent results
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# 1) Function to generate random planet parameters (improved)
def generate_planet(habitable=0):
    if habitable == 1:
        orbital_distance = random.uniform(0.95, 1.05)
        planet_radius = random.uniform(0.9, 1.1)
        surface_temp = random.uniform(270, 300)
    else:
        choices = [
            (random.uniform(0.01, 0.7), random.uniform(0.1, 0.8), random.uniform(100, 250)),  # cold planets
            (random.uniform(1.2, 15.0), random.uniform(0.8, 10.0), random.uniform(350, 1500)) # hot/far planets
        ]
        orbital_distance, planet_radius, surface_temp = random.choice(choices)
    return [orbital_distance, planet_radius, surface_temp, habitable]

# 2) Generate synthetic dataset
planet_data = []
for _ in range(150):
    planet_data.append(generate_planet(habitable=1 if random.random() < 0.4 else 0))

# 3) Add real planet examples
real_data = pd.DataFrame([
    [1.0, 1.0, 288, 1],     # Earth
    [1.52, 0.53, 210, 0],   # Mars
    [0.39, 0.38, 700, 0],   # Mercury
    [0.72, 0.95, 737, 0]    # Venus
], columns=['Orbital_Distance_AU', 'Planet_Radius_Earth', 'Surface_Temperature_K', 'Habitable'])

planet_data.extend(real_data.values.tolist())

# 4) Convert to dataframe
data = pd.DataFrame(planet_data, columns=['Orbital_Distance_AU', 'Planet_Radius_Earth', 'Surface_Temperature_K', 'Habitable'])

# Optional: Visualize distribution
# sns.pairplot(data, hue='Habitable')
# plt.suptitle("Data distribution", y=1.02)
# plt.show()

# 5) Feature selection
X = data[['Orbital_Distance_AU', 'Planet_Radius_Earth', 'Surface_Temperature_K']]
Y = data['Habitable']

# 6) Encode target
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

# 7) Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# 8) Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9) Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# 10) Build model with dropout
model = Sequential([
    Dense(32, activation="relu", input_shape=(3,)),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# 11) Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 12) Train model
history = model.fit(X_train, Y_train, epochs=150, batch_size=10, validation_data=(X_test, Y_test), class_weight=class_weight_dict)

# 13) Evaluate
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"\n✅ Accuracy on test set: {accuracy * 100:.1f} %")

# 14) Predict on real planets
custom_planets = pd.DataFrame([
    [1.0, 1.0, 288],     # Earth
    [1.52, 0.53, 210],   # Mars
    [0.39, 0.38, 700],   # Mercury
    [0.72, 0.95, 737]    # Venus
], columns=X.columns)

custom_scaled = scaler.transform(custom_planets)
predictions = model.predict(custom_scaled)

planet_names = ["Earth", "Mars", "Mercury", "Venus"]
print()
for i, pred in enumerate(predictions):
    status = "Habitable" if pred >= 0.5 else "Not Habitable"
    print(f"{planet_names[i]}: {status} (Probability: {pred[0]:.2f})")

# 15) Plot training performance
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='red')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
