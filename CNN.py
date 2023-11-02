import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

#grab data from folder (creating a data pipeline)
    #puts data in batches of 32, makes all pictures same size, and adds the label
data = tf.keras.utils.image_dataset_from_directory('training_data')
data = data.map(lambda x,y: (x/255, y))
test_data = tf.keras.utils.image_dataset_from_directory('testing_data')
test_data = test_data.map(lambda x,y: (x/255, y))

#create a validation data set
train_size = int(len(data)*.8)
val_size = int(len(data)*.2)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)


#create the CNN model
model = Sequential()
#16 filters, size 3x3, stride 1, input is 256 by 256 3 pixels deep
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
#grab the max value
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
#convert to a 1d tensor
model.add(Flatten())
#send to 256 neurons
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
#1 layer
model.add(Dense(1, activation='sigmoid'))
#use optimizer - adam and loss - BinaryCrossEntropy
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

#Train the model
hist = model.fit(train, epochs=20, validation_data=val)

#set the precision, recall, and accuracy
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()
#testing the data
for newBatch in test_data.as_numpy_iterator(): 
    X, y = newBatch
    yhat = model.predict(X) #predict the outcome
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)

print(f'precision:{precision.result().numpy()} recall:{recall.result().numpy()} accuracy{accuracy.result().numpy()}')