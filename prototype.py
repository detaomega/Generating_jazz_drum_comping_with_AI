import dataset
import numpy as np
from tensorflow.keras import layers, models


X, Y = dataset.prepare('test/midi')

model = models.Sequential()
model.add(layers.Conv1D(filters=32, kernel_size=32, activation='relu', input_shape=(X[0].shape[0], X[0].shape[1])))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(Y[0].shape[0], activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()
model.fit(X, Y, epochs=5, verbose=1)
model.save("test/saved_model")



