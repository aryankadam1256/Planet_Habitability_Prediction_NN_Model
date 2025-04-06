import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv('newdataset.csv')  # New dataset with exoplanets & moons

# Encode 'Star_Type'
label_encoder = LabelEncoder()
data['Star_Type'] = label_encoder.fit_transform(data['Star_Type'])

# Select features and target
X = data[['Orbital_Distance_AU', 'Surface_Temperature_K', 'Planet_Radius_Earth', 'Star_Type', 'Surface_Gravity_g']]
y = data['Habitable']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a deeper neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # More neurons
    Dense(64, activation='relu'),  # Added depth
    Dense(32, activation='relu'),  # More complexity
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=500, batch_size=16, verbose=1)

# Custom input (planets & moons)
custom_input = np.array([
    [1.00, 288, 1.00, 0, 9.80],   # Earth
    [0.39, 440, 0.38, 0, 3.70],   # Mercury
    [1.52, 210, 0.53, 0, 3.70],   # Mars
    [5.20, 165, 11.21, 0, 24.80],  # Jupiter
    [39.48, 55, 0.18, 0, 0.62],   # Pluto
    [0.671, 102, 0.245, 0, 1.31],  # Europa
    [1.222, 94, 0.404, 0, 1.35],   # Titan
    [0.0013, 273, 0.04, 0, 0.11]   # Enceladus
])

# Scale custom input
custom_input_scaled = scaler.transform(custom_input)

# Make predictions
predictions = model.predict(custom_input_scaled)

# Print results
planet_names = ["Earth", "Mercury", "Mars", "Jupiter", "Pluto", "Europa", "Titan", "Enceladus"]

for i in range(len(predictions)):
    habitability_score = predictions[i][0] * 100
    print(f"Planet/Moon: {planet_names[i]}")
    print(f"  Habitability Score: {habitability_score:.2f}%")
    if habitability_score > 50:
        print("  --> Possibly Habitable\n")
    elif habitability_score > 20:
        print("  --> Has Some Potential\n")
    else:
        print("  --> Not Habitable\n")















# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# # Load the dataset
# data = pd.read_csv('newdataset.csv')

# # Encode 'Star_Type'
# label_encoder = LabelEncoder()
# data['Star_Type'] = label_encoder.fit_transform(data['Star_Type'])

# # Select features and target
# X = data[['Orbital_Distance_AU', 'Surface_Temperature_K', 'Planet_Radius_Earth', 'Star_Type', 'Surface_Gravity_g']]
# y = data['Habitable']

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Build model
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Increased units
#     Dense(32, activation='relu'),  # Added a layer
#     Dense(1, activation='sigmoid')
# ])

# # Compile model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train model
# model.fit(X_train, y_train, epochs=300, batch_size=10)# verbose=0, validation_split=0.1) # Increased epochs and batch size

# # Custom input
# custom_input = np.array([
#     [1.00, 288, 1.00, 0, 9.80],   # Earth
#     [0.39, 440, 0.38, 0, 3.70],   # Mercury
#     [1.52, 210, 0.53, 0, 3.70],   # Mars
#     [5.20, 165, 11.21, 0, 24.80],  # Jupiter
#     [39.48, 55, 0.18, 0, 0.62],   # Pluto
# ])

# # Scale custom input
# custom_input_scaled = scaler.transform(custom_input)

# # Make predictions
# predictions = model.predict(custom_input_scaled)

# # Print results
# for i in range(len(predictions)):
#     # print("Habitability: {:.2f}".format(predictions[i][0]))
#     print("the habitability of this planet is ",predictions[i][0]*100,)
#     if predictions[i][0] > 0.5:
#         print("--> HABITABLE\n")
#     else:
#         print("--> NOT HABITABLE\n")



















# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# data = pd.read_csv('newdataset.csv')  

# label_encoder = LabelEncoder()
# data['Star_Type'] = label_encoder.fit_transform(data['Star_Type'])

# X=data[['Orbital_Distance_AU','Surface_Temperature_K','Planet_Radius_Earth','Star_Type','Surface_Gravity_g']]

# y=data['Habitable'];

# scaler=StandardScaler()  
# X=scaler.fit_transform(X) # we normalize values to avoid overfitting or 1 paramter dominating other 

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# model=Sequential([
#     Dense(units=5,activation='relu',input_shape=(5,1)), # this is the only hidden layer which follows basic linear reg then relu for non linearity
#     Dense(units=1,activation='sigmoid') # this is output layer
# ])

# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# # model.fit(X_train,y_train,epochs=150,batch_size=5)
# model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=0, validation_split=0.1)


# # manual_input=np.array[[]]
# custom_input = np.array([
#     [0.39, 440, 0.38, 1, 3.7],   # Mercury
#     [1.00, 288, 1.00, 1, 9.8],   # Earth
#     [1.52, 210, 0.53, 1, 3.7],   # Mars
#     [5.20, 165, 11.21, 1, 24.8], # Jupiter
#     [39.48, 55, 0.18, 1, 0.62]   # Pluto
# ])
# custom_input=scaler.transform(custom_input)
# prediction=model.predict(custom_input)
# feature_labels = ["Orbital Distance (AU)", "Surface Temp (°C)", "Planetary Radius (Earth=1)", "Star Type", "Surface Gravity (m/s²)"]

# planet_names = ["Mercury", "Earth", "Mars", "Jupiter", "Pluto"] # planet names

# # for name, values in zip(planet_names, custom_input):
# #     print(f"\n{name}:")
# #     for label, value in zip(feature_labels, values):
# #         print(f"  {label}: {value}")

# # # first_sample_feature_labels=data[['Orbital_Distance_AU','Surface_Temperature_K','Planet_Radius_Earth','Star_Type','Surface_Gravity_g']].iloc[0]
# # # print(first_sample_feature_labels)
# # # Prediction for 1st test sample in dataset is 
# # print("the predicted habitability is ",prediction[0][0]*100,"%")
# # if prediction[0][0] > 0.5 : # we chekc for value of 1st test sample 
# #     print("the planet is HABITABLE")
# # else:
# #     print("the planet is NOT HABITABLE")

# for i in range(len(planet_names)):  # Loop over each planet
#     print(f"Planet Name: {planet_names[i]}")
    
#     for j in range(len(feature_labels)):  # Loop over each feature
#         print(f"  {feature_labels[j]}: {custom_input[i][j]}")
    
#     # Print habitability below its respective information
#     print("HABITABILITY is ",prediction[i][0]*100,"%")
#     if prediction[i][0] > 0.5:
#         print("  → This planet is **HABITABLE**\n")
#     else:
#         print("  → This planet is **NOT HABITABLE**\n")