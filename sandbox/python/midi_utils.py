import numpy as np
import pretty_midi
from keras.utils import to_categorical

def parse_midi(path):
    midi = None
    with open(path, 'r') as f:
        try:
            midi = pretty_midi.PrettyMIDI(f)
            midi.remove_invalid_notes()
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

# create a pretty midi file with a single instrument using the one-hot encoding
# output of keras model.predict.
def network_output_to_midi(windows, 
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

# create a pretty midi file with a single instrument using the one-hot encoding
# output of keras model.predict.
def network_output_intervals_to_midi(windows,
                                     start_note,
                                     instrument_name='Acoustic Grand Piano', 
                                     allow_represses=False):

    def from_one_hot(vec, rest_token=1000):
        index = np.argmax(vec)
        if index == 0:
            return rest_token
        else:
            # weirdly this has to be mapped from 0-99 if to_one_hot is mapping
            # from 1-100
            return map_range((0, 99), (-50, 50), index)

    def map_range(a, b, s):
        (a1, a2), (b1, b2) = a, b
        return  b1 + ((s - a1) * (b2 - b1) / (a2 - a1))

    def clamp(val, min_, max_):
        return min_ if val < min_ else max_ if val > max_ else val

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a cello instrument
    instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=instrument_program)
    
    cur_note = None # an invalid note to start with
    cur_note_start = None
    clock = 0

    last_played_note = start_note

    # Iterate over note names, which will be converted to note number later
    for step in windows:

        interval = from_one_hot(step)
        if interval == 1000:
            note_num = -1
        else:
            last_played_note = clamp(last_played_note + interval, 0, 127)
            note_num = last_played_note
        
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
