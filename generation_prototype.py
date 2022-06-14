from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
import pretty_midi
import dataset
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

model = models.load_model("test/saved_model")
file = 'test/midi/autumn_leaves_pt_dm.mid'
stream = pretty_midi.PrettyMIDI(file)
notdrum, drum = dataset.midi_to_piano_roll(file, version=2)
roll_length = drum.shape[0]

X = []
Y = []

i = 0
while i < (roll_length // 512):
    X.append(notdrum[i * 512 : (i + 1) * 512])
    i += 1
X = np.array(X) #製作輸入

prediction = model.predict(X, verbose=0)
for output in prediction:
    tmp = []
    for i in range(512):
        tmp.append(output[i * 2 : (i + 1) * 2])
    Y.append(tmp)
Y = np.array(Y)

tmp = []
for M in Y:
    for row in M:
        tmp.append(row)
Y = np.array(tmp).transpose() #將輸出轉成橫的piano roll

cnt = 0
tmp = []
for row in Y:
    for i in row:
        if i > 0:
            cnt += 1
            tmp.append(i)
tmp = np.array(tmp)
Y = np.where(Y > tmp.mean() + tmp.std() * 2, 1, 0) #決定哪些要留

for row in Y:
    i = 0
    while i < len(row):
        if row[i] == 0:
            j = 0
            while i + j < len(row) and row[i + j] == 0:
                j += 1
            if j <= 64:
                k = i
                while k < i + j:
                    row[k] = 1
                    k += 1
            i += j
        else:
            i += 1 #將一些空格填滿


drum = drum.transpose()
for i in range(128):
    if i == 36:
        for j in range(len(Y[0])):
            drum[i][j] = Y[0][j]
    elif i == 40:
        for j in range(len(Y[1])):
            drum[i][j] = Y[1][j] #把大鼓跟小鼓貼回原本的檔案
drum = np.where(drum > 0, 80, 0)
bpm = stream.estimate_tempo()
newdrum = dataset.piano_roll_to_instrument(drum, fs=(bpm * 128) / 60)
newdrum.is_drum = True
newInstrument = []
for instrument in stream.instruments:
    if not instrument.is_drum:
        newInstrument.append(instrument)
newInstrument.append(newdrum)
stream.instruments = newInstrument
stream.write('test.mid')