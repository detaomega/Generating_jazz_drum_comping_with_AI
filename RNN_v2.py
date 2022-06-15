import dataset
import numpy as np
from tensorflow.keras import layers, models
import pickle


X, Y = dataset.prepare('midi/smaller/training')
X_val, Y_val = dataset.prepare('midi/smaller/validation')


model = models.Sequential()
model.add(
    layers.LSTM(
    512,
    input_shape=(X[0].shape[0], X[0].shape[1]),
    return_sequences=True
))
model.add(layers.Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(Y[0].shape[0], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X, Y, epochs=5, verbose=1, validation_data=(X_val, Y_val))
model.save("test/saved_model")
with open('test/1sttrainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


