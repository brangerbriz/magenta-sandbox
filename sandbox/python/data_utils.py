import pdb
import midi_utils
import numpy as np
from multiprocessing import Pool as ThreadPool
from keras.utils.np_utils import to_categorical

#format enum
F_VANILLA_WINDOW       = 1
F_EVENT_WINDOW_NOTES   = 2
F_EVENT_WINDOW_TIMES   = 3

# average windows per file (calculated using a window size of 
# 20 and ~5,000 midi files). Useful in approximating an epoch 
# size before data is loaded (like w/ generators).
WINDOWS_PER_FILE = 824 

# loads everything into RAM at once
def get_data(midi_paths, 
             form=F_VANILLA_WINDOW, 
             window_size=20, 
             num_threads=8):
        pass

# load data with a lazzy loader
def get_data_generator(midi_paths, 
                       form=F_VANILLA_WINDOW, 
                       window_size=20, 
                       num_threads=8,
                       batch_size=32):

    def extract_data(pm_midi, form, window_size):

        if form == F_VANILLA_WINDOW or \
           form == F_EVENT_WINDOW_NOTES or \
           form == F_EVENT_WINDOW_TIMES:
            return windows_from_monophonic_instruments(pm_midi, form, window_size)
        else:
            raise('Unsupported form parameter')

    load_size = 400 # this refers to files

     # load midi data
    pool = ThreadPool(num_threads)

    load_index = 0

    while True:
        
        load_files = midi_paths[load_index:load_index + load_size]
        # print('length of load files: {}'.format(len(load_files)))
        load_index = (load_index + load_size) % len(midi_paths)

        # print('loading large batch: {}'.format(load_size))
        # print('Parsing midi files...')
        # start_time = time.time()
        parsed = pool.map(midi_utils.parse_midi, load_files)
        # print('Finished in {:.2f} seconds'.format(time.time() - start_time))
        # print('parsed, now extracting data')
        data = extract_data(parsed, form, window_size)
        batch_index = 0
        while batch_index + batch_size < len(data[0]):
            # print('getting data...')
            # print('yielding small batch: {}'.format(batch_size))
            
            res = (data[0][batch_index: batch_index + batch_size], 
                   data[1][batch_index: batch_index + batch_size])
            yield res
            batch_index = batch_index + batch_size
        
        del parsed # free the mem
        del data # free the mem


# returns X, y data windows from all monophonic instrument
# tracks in a pretty midi file
def windows_from_monophonic_instruments(midi, form, window_size):
    X, y = [], []
    for m in midi:
        if m is not None:
            melody_instruments = midi_utils.filter_monophonic(m.instruments, 
                                                              1.0)
            for instrument in melody_instruments:
                if len(instrument.notes) > window_size:
                    windows = None
                    if form == F_VANILLA_WINDOW:
                        windows = _encode_sliding_windows(instrument, window_size)
                    elif form == F_EVENT_WINDOW_NOTES:
                        windows = _encode_sliding_window_notes(instrument, window_size)
                    elif form == F_EVENT_WINDOW_TIMES:
                        windows = _encode_sliding_window_times(instrument, window_size)
                    for w in windows:
                        X.append(w[0])
                        y.append(w[1])
    return (np.asarray(X), np.asarray(y))

# one-hot encode a sliding window of notes from a pretty midi instrument.
# This approach encodes note pitches only and does not contain timing 
# information or rests.
# expects pm_instrument to be monophonic.
def _encode_sliding_window_notes(pm_instrument, window_size, num_classes=128):
    notes = [n.pitch for n in pm_instrument.notes]
    windows = []
    for i in range(0, len(notes) - window_size - 1):
        window = ([to_categorical(n, num_classes=num_classes).flatten() for n in notes[i:i + window_size]], 
                        to_categorical(notes[i + window_size + 1], num_classes=num_classes).flatten())
        windows.append(window)
    return windows

# COME BACK HERE ON MONDAY
def _encode_sliding_window_times(pm_instrument, window_size):
    notes = [(n.start, n.end) for n in pm_instrument.notes]
    windows = []
    # windows.append((0.0, 0.0))
    for i in range(1, len(notes) - window_size - 1):
        x = notes[i:i + window_size]
        y = notes[i + window_size + 1]
        prev_x = notes[i - 1:i + window_size - 1]
        prev_y = notes[i + window_size]  

        # this fails ocassionally: y[0] < prev_y[0]
        # and this more so does this: y[0] < prev_y[1]
        # if y[0] < prev_y[0]:
        #     print('fuck 1')
        # if y[0] < prev_y[1]:
        #     print('fuck 2')

        X = []
        # create (offset, duration) pairs, where the first element is the offset
        # of this note from the last note off and the second element is the 
        # length of this note. Both values expressed in seconds. 
        for j in range(window_size):
            # note on minus last note off
            offset   = x[j][0] - prev_x[j][1]
            # note off minus this note on
            duration = x[j][1] - x[j][0]
            
            # if offset < 0 or duration < 0:
            #     pdb.set_trace()

            X.append([offset, duration])    

        # note on minus last note off, note off minus this note on
        Y = [y[0] - prev_y[1], y[1] - y[0]]
        windows.append((X, Y))
    return windows

# one-hot encode a sliding window of notes from a pretty midi instrument.
# This approach uses the piano roll method, where each step in the sliding
# window represents a constant unit of time (fs=4, or 1 sec / 4 = 250ms).
# This allows us to encode rests.
# expects pm_instrument to be monophonic.
def _encode_sliding_windows(pm_instrument, window_size):
    
    roll = np.copy(pm_instrument.get_piano_roll(fs=4).T)

    # trim beginning silence
    summed = np.sum(roll, axis=1)
    mask = (summed > 0).astype(float)
    roll = roll[np.argmax(mask):]
    
    # transform note velocities into 1s
    roll = (roll > 0).astype(float)
    
    # calculate the percentage of the events that are rests
    s = np.sum(roll, axis=1)
    num_silence = len(np.where(s == 0)[0])
    # print('{}/{} {:.2f} events are rests'.format(num_silence, len(roll), float(num_silence)/float(len(roll))))

    # append a feature: 1 to rests and 0 to notes
    rests = np.sum(roll, axis=1)
    rests = (rests != 1).astype(float)
    roll = np.insert(roll, 0, rests, axis=1)
    
    windows = []
    for i in range(0, roll.shape[0] - window_size - 1):
        windows.append((roll[i:i + window_size], roll[i + window_size + 1]))
    return windows
