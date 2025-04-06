import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1) firstly we will read the dataset 
data=pd.read_csv("dataset_3.csv")

# 2) here we will select the features from the dataset as x and y 
X=data[['Orbital_Distance_AU','Planet_Radius_Earth','Surface_Temperature_K']]
Y=data['Habitable'] # this is our target

# 3) once selected or initialized teh variables noew we will split the data for the  the model TESTING and TRAINING
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42) # here we allot 20% for testing and 80% for training

# 4) now we have to scale & normalize trained the data
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test) # we use the same scaling for test data which we used for training to avoid inconsistencies as fit_transform on test data will result some

# 5) now we build a neural network

model=Sequential([
    Dense(4,activation="relu",input_shape=(3,)), # hidden layer where we have taken 4 neurons and input 3 paramters
    Dense(1,activation='sigmoid') # output :- logistic 0 or 1
])

# 6) compille the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

# 7) the model is now trained using fit(), then we evaluate to calcualte the loss and accruacy
model.fit(X_train,Y_train,epochs=50,batch_size=10,validation_data=(X_test,Y_test))

loss,accuracy=model.evaluate(X_test,Y_test)
print("the accuracy of the model is : ",accuracy,"%")

# 8) we give custom inputs to the model and the scale them
new_planets = np.array([
    [1.0, 1.0, 288],  # Earth (expected: habitable)
    [1.52, 0.53, 210], # Mars (expected: non-habitable)
    [0.39, 0.38, 700], # Mercury (expected: non-habitable)
    [0.72, 0.95, 737]  # Venus (expected: non-habitable)
])

new_planets=scaler.transform(new_planets)

# 9) we run the predictions for final model output 

predictions=model.predict(new_planets)

# 10) now we print the predictions for the custom input data
for i, pred in enumerate(predictions):
    status = "Habitable" if pred >= 0.5 else "Not Habitable"
    print(f"Planet {i+1}: {status} (Probability: {pred[0]:.2f})")




