import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and clean the dataset
data = pd.read_csv('exoplanet_habitability_test_2_parameters.csv')
data = data.dropna(axis=1, how='all')  # Remove empty columns from CSV
data = data.dropna()

# Select features and target
X = data[['Orbital_Distance_AU', 'Planet_Radius_Earth', 'Surface_Temperature_K']]
y = data['Habitable']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build simplified model
model = Sequential([
    Dense(8, activation='relu', input_shape=(3,)),  # Input for 3 parameters
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=150, batch_size=8, validation_split=0.2, verbose = 0)

# Custom input for 3 parameters
custom_input = np.array([
    [1.00, 1.00, 288],   # Earth-like
    [0.39, 0.38, 440],   # Mercury-like
    [1.52, 0.53, 210],   # Mars-like
])

custom_input_scaled = scaler.transform(custom_input)
predictions = model.predict(custom_input_scaled)

# Print results
planet_names = ["Earth-like", "Mercury-like", "Mars-like"]
for i in range(len(planet_names)):
    print(f"Planet: {planet_names[i]}")
    print(f"  Habitability: {predictions[i][0]*100:.2f}%")
    print("  Status: HABITABLE" if predictions[i][0] > 0.5 else "  Status: NOT HABITABLE\n")
