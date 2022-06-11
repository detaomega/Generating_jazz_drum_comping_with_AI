from typing import Dict
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

version = 1
stream = pretty_midi.PrettyMIDI('test/midi/fly_me_to_the_moon-Frank-Sinatra-kar_mm.mid')
bpm = stream.estimate_tempo()
fs = (bpm * 128) / 60
# find the max column in instruments
max_col = -1
for i in range(len(stream.instruments)):
    piano_roll = my_get_piano_roll(stream.instruments[i], fs)
    row, col = piano_roll.shape
    max_col = max(max_col, col)

# version 1 (row: instrument, col: time, velocity is added together)
not_drum_instruments = np.zeros((128, max_col))
drum_instruments = np.zeros((128, max_col))

for i in range(len(stream.instruments)):
    piano_roll = my_get_piano_roll(stream.instruments[i], fs)
    row, col = piano_roll.shape
    if stream.instruments[i].is_drum == False:
        tem = np.hstack((piano_roll, np.zeros((128, max_col - col))))
        not_drum_instruments += tem
    else :
        tem = np.hstack((piano_roll, np.zeros((128, max_col - col))))
        drum_instruments += tem

# version 2 (row: time, col: instrument, changed to zeros and ones)
if version == 2:
    drum_instruments = np.where(drum_instruments > 0, 1, 0)
    drum_instruments = drum_instruments.transpose()

for row in drum_instruments:
    for i in row:
        print(i, end=' ')
    print("")