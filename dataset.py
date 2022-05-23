from typing import Dict
from mido import KeySignatureError
import pretty_midi
import numpy as np

def my_get_piano_roll(self, fs=100, times=None,
                    pedal_threshold=64):
    """Compute a piano roll matrix of this instrument.
    Parameters
    ----------
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    times : np.ndarray
        Times of the start of each column in the piano roll.
        Default ``None`` which is ``np.arange(0, get_end_time(), 1./fs)``.
    pedal_threshold : int
        Value of control change 64 (sustain pedal) message that is less
        than this value is reflected as pedal-off.  Pedals will be
        reflected as elongation of notes in the piano roll.
        If None, then CC64 message is ignored.
        Default is 64.
    Returns
    -------
    piano_roll : np.ndarray, shape=(128,times.shape[0])
        Piano roll of this instrument.
    """
    # If there are no notes, return an empty matrix
    if self.notes == []:
        return np.array([[]]*128)
    # Get the end time of the last event
    end_time = self.get_end_time()
    # Extend end time if one was provided
    if times is not None and times[-1] > end_time:
        end_time = times[-1]
    # Allocate a matrix of zeros - we will add in as we go
    piano_roll = np.zeros((128, int(fs*end_time)))
    # Drum tracks don't have pitch, so return a matrix of zeros
    # if self.is_drum:
    #     if times is None:
    #         return piano_roll
    #     else:
    #         return np.zeros((128, times.shape[0]))
    # Add up piano roll matrix, note-by-note
    for note in self.notes:
        # Should interpolate
        piano_roll[note.pitch, int(note.start*fs):int(note.end*fs)] += note.velocity

    if times is None:
        return piano_roll
    piano_roll_integrated = np.zeros((128, times.shape[0]))
    # Convert to column indices
    times = np.array(np.round(times*fs), dtype=np.int32)
    for n, (start, end) in enumerate(zip(times[:-1], times[1:])):
        if start < piano_roll.shape[1]:  # if start is >=, leave zeros
            if start == end:
                end = start + 1
            # Each column is the mean of the columns in piano_roll
            piano_roll_integrated[:, n] = np.mean(piano_roll[:, start:end], axis=1)
    return piano_roll_integrated

stream = pretty_midi.PrettyMIDI('fly_me_to_the_moon-Frank-Sinatra-kar_mm.mid')


# find the max column in instruments
max_col = -1
for i in range(len(stream.instruments)):
    row, col = my_get_piano_roll(stream.instruments[i]).shape
    max_col = max(max_col, col)

# version 1
not_drum_instruments_v1 = np.zeros((128, max_col))
drum_instruments_v1 = np.zeros((128, max_col))

for i in range(len(stream.instruments)):
    row, col = my_get_piano_roll(stream.instruments[i]).shape
    if stream.instruments[i].is_drum == False:
        tem = np.hstack((my_get_piano_roll(stream.instruments[i]), np.zeros((128, max_col - col))))
        not_drum_instruments_v1 += tem
    else :
        tem = np.hstack((my_get_piano_roll(stream.instruments[i]), np.zeros((128, max_col - col))))
        drum_instruments_v1 += tem

# version 2
drum_instruments_v2 = np.zeros((128, max_col))

for i in range(len(stream.instruments)):
    row, col = my_get_piano_roll(stream.instruments[i]).shape
    if stream.instruments[i].is_drum:
        tem = np.hstack((my_get_piano_roll(stream.instruments[i]), np.zeros((128, max_col - col))))
        drum_instruments_v2 += tem

drum_instruments_v2 = np.where(drum_instruments_v2 > 0, 1, 0)
drum_instruments_v2 = drum_instruments_v2.transpose()

for row in drum_instruments_v2:
    for i in row:
        print(i, end=' ')
    print("")