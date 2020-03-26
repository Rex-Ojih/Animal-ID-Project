import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import pickle as pk
import numpy as np

pickle_in = open("X-features.pk", "rb")
X = pk.load(pickle_in) #load in features (images) 
pickle_in.close()

pickle_in = open("y-labels.pk", "rb")
y = pk.load(pickle_in) #load in features (images) 
pickle_in.close()

y = np.array(y).reshape(-1, 1)

X = X/255.0 #normalizing feature pixel data to be between 0 and 1

model = Sequential()

model.add(Conv2D(128, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(10))
model.add(Activation("softmax"))
 
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X, y, batch_size=64, validation_split=0.1, epochs=5)
