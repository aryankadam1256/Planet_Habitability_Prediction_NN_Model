# Planet_Habitability_Prediction_NN_Model
This is a basic NN model which uses single hidden layer , it predicts whether a Planet with any input parameters is Habitable for life.

🌍 Habitable Planet Classifier – README Explanation
This project demonstrates how to build a binary classification model using TensorFlow and Keras to determine whether a planet is potentially habitable based on its parameters like orbital distance, planet radius, and surface temperature. It combines synthetic data with real-world planet data (Earth, Mars, etc.) to create a simple yet effective learning system.

📦 1. Importing Required Libraries
python
Copy
Edit
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
Standard libraries like numpy, pandas, matplotlib, and seaborn are used for data handling and visualization.

scikit-learn is used for data preprocessing and evaluation support.

tensorflow.keras is used for building and training a neural network.

random is used for generating synthetic values.

🔒 2. Setting Random Seeds
python
Copy
Edit
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
This ensures reproducibility by fixing random seed values for Python, NumPy, and TensorFlow.

🪐 3. Synthetic Planet Data Generation
python
Copy
Edit
def generate_planet(habitable=0):
    if habitable == 1:
        orbital_distance = random.uniform(0.95, 1.05)
        planet_radius = random.uniform(0.9, 1.1)
        surface_temp = random.uniform(270, 300)
    else:
        choices = [
            (random.uniform(0.01, 0.7), random.uniform(0.1, 0.8), random.uniform(100, 250)),
            (random.uniform(1.2, 15.0), random.uniform(0.8, 10.0), random.uniform(350, 1500))
        ]
        orbital_distance, planet_radius, surface_temp = random.choice(choices)
    return [orbital_distance, planet_radius, surface_temp, habitable]
This function simulates planets.

Habitable planets have Earth-like values (Goldilocks zone, moderate radius, optimal temperature).

Non-habitable planets are either too cold (e.g., far from their star) or too hot (e.g., very close).

🧪 4. Dataset Creation
python
Copy
Edit
planet_data = []
for _ in range(150):
    planet_data.append(generate_planet(habitable=1 if random.random() < 0.4 else 0))
Generates 150 synthetic planets, with ~40% being habitable.

🌍 5. Adding Real Planet Examples
python
Copy
Edit
real_data = pd.DataFrame([
    [1.0, 1.0, 288, 1],     # Earth
    [1.52, 0.53, 210, 0],   # Mars
    [0.39, 0.38, 700, 0],   # Mercury
    [0.72, 0.95, 737, 0]    # Venus
], columns=['Orbital_Distance_AU', 'Planet_Radius_Earth', 'Surface_Temperature_K', 'Habitable'])
Includes real-world planet data for Earth (habitable) and others (not habitable) to enhance realism and grounding.

🧱 6. DataFrame Conversion
python
Copy
Edit
planet_data.extend(real_data.values.tolist())
data = pd.DataFrame(planet_data, columns=['Orbital_Distance_AU', 'Planet_Radius_Earth', 'Surface_Temperature_K', 'Habitable'])
Combines synthetic and real data into a single DataFrame.

📊 7. Feature & Label Selection
python
Copy
Edit
X = data[['Orbital_Distance_AU', 'Planet_Radius_Earth', 'Surface_Temperature_K']]
Y = data['Habitable']
X: input features.

Y: target label (0 or 1).

🔢 8. Label Encoding
python
Copy
Edit
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
Converts the target labels to numerical format (although it’s already binary, this ensures robustness).

🧪 9. Train/Test Split
python
Copy
Edit
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
Splits data: 80% training, 20% testing.

stratify=Y ensures class balance in both sets.

⚖️ 10. Feature Scaling
python
Copy
Edit
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Scales features between 0 and 1 to improve neural network performance.

⚖️ 11. Class Weight Calculation
python
Copy
Edit
class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
Computes class weights to handle imbalance in the dataset (more non-habitable planets than habitable).

🧠 12. Neural Network Architecture
python
Copy
Edit
model = Sequential([
    Dense(32, activation="relu", input_shape=(3,)),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
A 3-layer feed-forward neural network:

First hidden layer: 32 neurons + ReLU + Dropout

Second hidden layer: 16 neurons + ReLU + Dropout

Output layer: 1 neuron with Sigmoid (binary classification)

⚙️ 13. Model Compilation
python
Copy
Edit
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
Uses Adam optimizer and binary crossentropy loss, suitable for binary classification.

📈 14. Model Training
python
Copy
Edit
history = model.fit(X_train, Y_train, epochs=150, batch_size=10, validation_data=(X_test, Y_test), class_weight=class_weight_dict)
Trains the model for 150 epochs.

Uses validation set to monitor overfitting.

Applies class weights to give importance to underrepresented class.

🧪 15. Model Evaluation
python
Copy
Edit
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"\n✅ Accuracy on test set: {accuracy * 100:.1f} %")
Evaluates model performance on the test set.

🔭 16. Custom Predictions
python
Copy
Edit
custom_planets = pd.DataFrame([
    [1.0, 1.0, 288],     # Earth
    [1.52, 0.53, 210],   # Mars
    [0.39, 0.38, 700],   # Mercury
    [0.72, 0.95, 737]    # Venus
], columns=X.columns)
Predicts habitability of real planets using the trained model.

python
Copy
Edit
custom_scaled = scaler.transform(custom_planets)
predictions = model.predict(custom_scaled)

planet_names = ["Earth", "Mars", "Mercury", "Venus"]
for i, pred in enumerate(predictions):
    status = "Habitable" if pred >= 0.5 else "Not Habitable"
    print(f"{planet_names[i]}: {status} (Probability: {pred[0]:.2f})")
Outputs probability and label (habitable or not).

📊 17. Training Performance Visualization
python
Copy
Edit
plt.figure(figsize=(12, 5))
...
plt.plot(...)
plt.title(...)
...
plt.tight_layout()
plt.show()
Plots loss and accuracy trends over epochs to visually inspect training and validation performance.

📌 Final Notes
This project is a simplified simulation that shows how machine learning can be applied to astrobiology concepts. While the data is synthetic and based on assumptions, it provides a good starting point for understanding classification tasks, neural networks, and model evaluation in Python.
