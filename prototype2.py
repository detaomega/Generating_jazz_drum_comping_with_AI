import dataset
import numpy as np
from tensorflow.keras import layers, models
import keras
import pickle


X, Y = dataset.prepare('test/midi/smaller/training')
X_val, Y_val = dataset.prepare('test/midi/smaller/validation')

model = models.Sequential()
model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(X[0].shape[0], X[0].shape[1], 1), padding="same"))
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(Y[0].shape[0], activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])
model.summary()
history = model.fit(X, Y, epochs=5, verbose=1, validation_data=(X_val, Y_val))
model.save("test/2nd_saved_model")
with open('test/2ndtrainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)



