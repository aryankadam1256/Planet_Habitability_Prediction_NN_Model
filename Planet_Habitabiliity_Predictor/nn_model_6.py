import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# 1) Read and clean dataset
data = pd.read_csv("dataset_6.csv").dropna()
data = data[(data['Surface_Temperature_K'] > 0) & 
           (data['Planet_Radius_Earth'] > 0) &
           (data['Orbital_Distance_AU'] > 0)]

# 2) Select features with clear habitable zone boundaries
X = data[['Orbital_Distance_AU', 'Planet_Radius_Earth', 'Surface_Temperature_K']]
Y = data['Habitable']

# 3) Enhanced data splitting with stratification
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=0.2, 
    random_state=42,
    stratify=Y  # Maintain class distribution
)

# 4) Improved scaling using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5) Dynamic class weighting
class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# 6) Enhanced model architecture
# model = Sequential([
#     Dense(32, activation="relu", 
#           input_shape=(3,), 
#           kernel_regularizer=l2(0.01)),
#     Dropout(0.3),
#     Dense(16, activation="relu"),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])
model = Sequential([
    Dense(6, activation="relu", kernel_regularizer=l2(0.01)),
    # Dropout(0.5),
    # Dense(8, activation="relu"),
    Dense(1, activation='sigmoid')
])

# 7) Optimized model compilation
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
            #  tensorflow.keras.metrics.Precision(name='precision'),
            #  tensorflow.keras.metrics.Recall(name='recall')]
)

# 8) Early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# 9) Enhanced training with validation
history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, Y_test),
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# 10) Comprehensive evaluation
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
# print(f"Precision: {test_precision*100:.2f}% | Recall: {test_recall*100:.2f}%")

# 11) Predict with confidence thresholds
new_planets = pd.DataFrame([
    [1.0, 1.0, 288],    # Earth (habitable)
    [0.39, 0.38, 700],  # Mercury (non-habitable)
    [1.52, 0.53, 210],  # Mars (non-habitable)
    [0.72, 0.95, 737],   # Venus (non-habitable)
], columns=X.columns)

new_planets_scaled = scaler.transform(new_planets)
predictions = model.predict(new_planets_scaled)

# 12) Clear prediction interpretation
for i, pred in enumerate(predictions):
    confidence = pred[0]
    if confidence >= 0.7:
        status = "HABITABLE (High Confidence)"
    elif confidence >= 0.5:
        status = "Potentially Habitable"
    elif confidence >= 0.3:
        status = "Marginally Habitable"
    else:
        status = "NOT HABITABLE"
    print(f"Planet {i+1}: {status} ({confidence*100:.2f}%)")

# 13) Enhanced visualization
plt.figure(figsize=(14, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Evolution')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Progress')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
