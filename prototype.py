import imp
import dataset
import train_model
import numpy as np
from tensorflow.keras import layers, models
import pickle

type = input("Which model tou want to train (CNN/RNN)?")
if type == "CNN":
    train_model.CNN()
elif type == "RNN":
    train_model.RNN()
else:
    print("wrong tpye")