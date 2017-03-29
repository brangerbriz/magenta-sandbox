import os, time, sys
sys.path.append('../../python')

import pdb, glob, random
import midi_utils, model_utils, data_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler

OUTPUT_SIZE = 128

# get the note model
def get_note_model(model_dir=None, window_size=20, model_index=0):
    
    epoch = 0
    
    if not model_dir:
        model = Sequential()
        model.add(LSTM(64,
                       return_sequences=False,
                       input_shape=(window_size, OUTPUT_SIZE)))
        model.add(Dropout(0.2))
        model.add(Dense(OUTPUT_SIZE))
        model.add(Activation('softmax'))
    else:
        model, epoch = model_utils.load_model_from_checkpoint(model_dir,
                                                              model_index=model_index)

    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    return model, epoch

# get the timing/rythm model
def get_timing_model(model_dir=None, window_size=20, model_index=1):
    
    epoch = 0
    
    if not model_dir:
        model = Sequential()
        
        # layer 1
        model.add(LSTM(34,
                       return_sequences=True,
                       input_shape=(window_size, 2)))
        model.add(Dropout(0.2))

        # layer 2
        model.add(LSTM(34,
                       return_sequences=False))
        model.add(Dropout(0.2))

        # layer 3
        model.add(Dense(2))
        model.add(Activation('linear'))

    else:
        model, epoch = model_utils.load_model_from_checkpoint(model_dir, 
                                                              model_index=model_index)

    model.compile(loss='mean_squared_error', 
                  optimizer='rmsprop',
                  metrics=['mse', 'mae', 'accuracy'])

    return model, epoch


# normalize the timing data using a MinMaxScaler
def scale_data(generator):

    scaler = MinMaxScaler()
    # how many batches?
    for i in range(1000):
        X, y = generator.next()
        for x in X:
            scaler.fit(x)

    while True:
        X, y = generator.next()
        for i, x in enumerate(range(len(X))):
            X[i] = scaler.transform(X[i]) 
        yield X, y

def main():

    midi_dir = '../../../data/query_symlinks'

    model_num = 1
    model_dir = '../../../models/keras/02_event_rnn/{}'.format(model_num)

    window_size = 20

    # only creates if doesn't already exist
    model_utils.create_model_dir(model_dir)

    files = [os.path.join(midi_dir, path) for path in os.listdir(midi_dir)]
    train_files = files[0:100]
    val_files   = files[100:120]

    # get the train/val d 
    train_generator = data_utils.get_data_generator(train_files, 
                                                    form=data_utils.F_EVENT_WINDOW_TIMES, 
                                                    window_size=20)

    val_generator = data_utils.get_data_generator(val_files,
                                                  form=data_utils.F_EVENT_WINDOW_TIMES, 
                                                  window_size=20)

    # train_generator = scale_data(train_generator)
    # val_generator = scale_data(val_generator)

    X_timing_seed, _ = val_generator.next()
    # pdb.set_trace()

    timing_model, epoch = get_timing_model(
                                           window_size=window_size, 
                                           model_index=1)

    note_model, _ = get_note_model(model_dir,
                                   window_size=window_size, 
                                   model_index=0)

    model_utils.save_model(note_model, model_dir, model_index=0)
    model_utils.save_model(timing_model, model_dir, model_index=1)
    
    callbacks = model_utils.get_callbacks(model_dir, 
                                          checkpoint_monitor='val_mae',
                                          model_index=1)
    
    print('fitting timing model...')
    timing_model.fit_generator(train_generator,
                               steps_per_epoch=data_utils.WINDOWS_PER_FILE * 10, 
                               epochs=10,
                               validation_data=val_generator, 
                               validation_steps=data_utils.WINDOWS_PER_FILE * 2,
                               verbose=1, 
                               callbacks=callbacks,
                               initial_epoch=epoch)

    train_generator = data_utils.get_data_generator(train_files, 
                                                    form=data_utils.F_EVENT_WINDOW_NOTES, 
                                                    window_size=20)


    callbacks = model_utils.get_callbacks(model_dir, 
                                          checkpoint_monitor='val_mae',
                                          model_index=0)

    val_generator = data_utils.get_data_generator(val_files,
                                                  form=data_utils.F_EVENT_WINDOW_NOTES, 
                                                  window_size=20)

    # print('fitting note model...')
    # note_model.fit_generator(train_generator,
    #                          steps_per_epoch=data_utils.WINDOWS_PER_FILE * 10, 
    #                          epochs=10,
    #                          validation_data=val_generator, 
    #                          validation_steps=data_utils.WINDOWS_PER_FILE * 2,
    #                          verbose=1, 
    #                          callbacks=callbacks,
    #                          initial_epoch=epoch)


    # generate 10 tracks using random seeds
    X_note_seed, _ = val_generator.next()

    print('generating notes...')
    generated_notes = model_utils.generate_notes(note_model, X_note_seed)
    
    print('generating timings...')
    # replace start/end note events with values generated from timing models
    generated_timings = model_utils.generate_timings(timing_model, X_timing_seed)

    for i, midi in enumerate(generated_notes):
        for instrument in midi.instruments:
            wall_time = 0
            for j, note in enumerate(instrument.notes):
                # print(i, j)
                # print(note)
                offset = generated_timings[i][j][0]
                duration = generated_timings[i][j][1]
                note.start = wall_time + offset
                note.end   = wall_time + offset + duration
                # print(note)
                print(generated_timings[i][j])
                # print('')
                wall_time = wall_time + offset + duration

    for i, midi in enumerate(generated_notes):
        file = os.path.join(model_dir, 'generated', '{}.mid'.format(i + 1))
        midi.write(file.format(i + 1))
        print('wrote midi file to {}'.format(file))
    
if __name__ == '__main__':
    main()