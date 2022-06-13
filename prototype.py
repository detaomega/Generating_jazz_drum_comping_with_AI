import dataset
import numpy as np
from tensorflow.keras import layers, models


file = 'test/midi/fly_me_to_the_moon-Frank-Sinatra-kar_mm.mid'
notdrum, drum = dataset.midi_to_piano_roll(file, version=2)
roll_length = drum.shape[0]

X = []
Y = []

i = 0
while i < (roll_length // 512):
    X.append(notdrum[i * 512 : (i + 1) * 512])
    tmp = []
    for row in drum[i * 512 : (i + 1) * 512]:
        for j in row:
            tmp.append(j)
    Y.append(tmp)
    i += 1

file = 'test/midi/autumn_leaves_pt_dm.mid'
notdrum, drum = dataset.midi_to_piano_roll(file, version=2)
roll_length = drum.shape[0]

i = 0
while i < (roll_length // 512):
    X.append(notdrum[i * 512 : (i + 1) * 512])
    tmp = []
    for row in drum[i * 512 : (i + 1) * 512]:
        for j in row:
            tmp.append(j)
    Y.append(tmp)
    i += 1

X = np.array(X)
Y = np.array(Y)

model = models.Sequential()
model.add(layers.Conv1D(filters=32, kernel_size=32, activation='relu', input_shape=(X[0].shape[0], X[0].shape[1])))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(Y[0].shape[0], activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()
model.fit(X, Y, epochs=5, verbose=1)
model.save("test/saved_model")



