import pickle
import matplotlib.pyplot as plt
history = pickle.load(open('RNNtrainHistoryDict', "rb"))
print(history.keys())
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')
plt.show()
