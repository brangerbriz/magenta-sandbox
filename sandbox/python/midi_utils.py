import numpy as np
import pretty_midi
from keras.utils import to_categorical

def parse_midi(path):
    midi = None
    with open(path, 'r') as f:
        try:
            midi = pretty_midi.PrettyMIDI(f)
        except:
            pass
    return midi

def get_percent_monophonic(pm_instrument_roll):
    mask = pm_instrument_roll.T > 0
    notes = np.sum(mask, axis=1)
    n = np.count_nonzero(notes)
    single = np.count_nonzero(notes == 1)
#     print('{}/{}'.format(single, n))
    if single > 0:
        return float(single) / float(n)
    elif single == 0 and n > 0:
        return 0.0
    else: # no notes of any kind
        return 0.0
    
def filter_monophonic(pm_instruments, percent_monophonic=0.99):
    return [i for i in pm_instruments if get_percent_monophonic(i.get_piano_roll()) >= percent_monophonic]

# returns a tuple of (family_index, family_name) for an instrument
# program number
# https://www.midi.org/specifications/item/gm-level-1-sound-set
def program_to_family(pgrm_num):
    if pgrm_num >=1 and pgrm_num <=8:
        return (0, 'Piano')
    elif pgrm_num >=9 and pgrm_num <=16:
        return (1, 'Chromatic Percussion')
    elif pgrm_num >= 17 and pgrm_num <= 24:
        return (2, 'Organ')
    elif pgrm_num >= 25 and pgrm_num <= 32:
        return (3, 'Guitar')
    elif pgrm_num >= 33 and pgrm_num <= 40:
        return (4, 'Bass')
    elif pgrm_num >= 41 and pgrm_num <= 48:
        return (5, 'Strings')
    elif pgrm_num >= 49 and pgrm_num <= 56:
        return (6, 'Ensemble')
    elif pgrm_num >= 57 and pgrm_num <= 64:
        return (7, 'Brass')
    elif pgrm_num >= 65 and pgrm_num <= 72:
        return (8, 'Reed')
    elif pgrm_num >= 73 and pgrm_num <= 80:
        return (9, 'Pipe')
    elif pgrm_num >= 81 and pgrm_num <= 88:
        return (10, 'Synth Lead')
    elif pgrm_num >= 89 and pgrm_num <= 96:
        return (11, 'Synth Pad')
    elif pgrm_num >= 97 and pgrm_num <= 104:
        return (12, 'Synth Effects')
    elif pgrm_num >= 105 and pgrm_num <= 112:
        return (13, 'Ethnic')
    elif pgrm_num >= 113 and pgrm_num <= 120:
        return (14, 'Percussive')
    elif pgrm_num >= 121 and pgrm_num <= 128:
        return (15, 'Sound Effects')
    else: # program number out of range
        return None

# one-hot encode a sliding window of notes from a pretty midi instrument.
# This approach encodes note pitches only and does not contain timing 
# information or rests.
# expects pm_instrument to be monophonic.
def encode_sliding_window_notes(pm_instrument, window_size=20):
    notes = [n.pitch for n in pm_instrument.notes]
    windows = []
    for i in range(0, len(notes) - window_size - 1):
        window = ([to_categorical(n, num_classes=128).flatten() for n in notes[i:i + window_size]], 
                        to_categorical(notes[i + window_size + 1], num_classes=128).flatten())
        windows.append(window)
    return windows

# one-hot encode a sliding window of notes from a pretty midi instrument.
# This approach uses the piano roll method, where each step in the sliding
# window represents a constant unit of time (fs=4, or 1 sec / 4 = 250ms).
# This allows us to encode rests.
# expects pm_instrument to be monophonic.
def encode_sliding_windows(pm_instrument, window_size=20):
    
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

# create a pretty midi file with a single instrument using the one-hot encoding
# output of keras model.predict.
def decode_sliding_windows(windows, 
                           instrument_name='Acoustic Grand Piano', 
                           allow_represses=False):

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a cello instrument
    instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=instrument_program)
    
    cur_note = None # an invalid note to start with
    cur_note_start = None
    clock = 0

    # Iterate over note names, which will be converted to note number later
    for step in windows:

        note_num = np.argmax(step) - 1
        
        # a note has changed
        if allow_represses or note_num != cur_note:
            
            # if a note has been played before and it wasn't a rest
            if cur_note is not None and cur_note >= 0:            
                # add the last note, now that we have its end time
                note = pretty_midi.Note(velocity=127, 
                                        pitch=int(cur_note), 
                                        start=cur_note_start, 
                                        end=clock)
                instrument.notes.append(note)

            # update the current note
            cur_note = note_num
            cur_note_start = clock

        # update the clock
        clock = clock + 1.0 / 4

    # Add the cello instrument to the PrettyMIDI object
    midi.instruments.append(instrument)
    return midi
