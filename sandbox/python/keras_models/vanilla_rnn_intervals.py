import os, time, sys
sys.path.append('../../python')

import pdb, glob, random
import midi_utils, model_utils, data_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam

OUTPUT_SIZE = 101
FORMAT = data_utils.F_VANILLA_WINDOW_INTERVALS

# create or load a saved model
# returns the model and the epoch number (>1 if loaded from checkpoint)
def get_model(model_dir=None, window_size=20):
    
    epoch = 0
    
    if not model_dir:
        model = Sequential()
        model.add(LSTM(64,
                       return_sequences=False,
                       input_shape=(window_size, OUTPUT_SIZE)))
        model.add(Dropout(0.5))
        # model.add(LSTM(32, return_sequences=False))
        # model.add(Dropout(0.2))
        model.add(Dense(OUTPUT_SIZE))
        model.add(Activation('softmax'))
    else:
        model, epoch = model_utils.load_model_from_checkpoint(model_dir)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    print(model, epoch)
    return model, epoch

def main():

    midi_dir = '../../../data/t-sne_selections/1/0/'

    model_num = 7 # same as 3
    model_dir = '../../../models/keras/' \
                '03_vanilla_rnn_intervals/{}'.format(model_num)

    window_size = 20

    # only creates if doesn't already exist
    model_utils.create_model_dir(model_dir)

    files = [os.path.join(midi_dir, path) for path in os.listdir(midi_dir)]

    val_split = 0.2
    print(int(float(len(files)) * val_split))
    train_generator = data_utils.get_data_generator(files[0:int(float(len(files)) * val_split)], 
                                                    form=FORMAT, 
                                                    window_size=20)

    val_generator = data_utils.get_data_generator(files[int(float(len(files)) * val_split):],
                                                  form=FORMAT, 
                                                  window_size=20)


    # intervals = []
    
    # for i in range(1000):
    #     data = train_generator.next()
    #     for window in data[0]:
    #         for interval in window:
    #             if interval not in intervals:
    #                 intervals.append(interval)

    # pdb.set_trace()

    # for i in range(1000):
    #     data = train_generator.next()
    #     pdb.set_trace()
    
    model, epoch = get_model(window_size=window_size)
    print(model.summary())
    model_utils.save_model(model, model_dir)
    
    callbacks = model_utils.get_callbacks(model_dir)
    
    # print('fitting model...')
    model.fit_generator(train_generator,
                        steps_per_epoch=1000, 
                        epochs=100,
                        validation_data=val_generator, 
                        validation_steps=200,
                        verbose=1, 
                        callbacks=callbacks,
                        initial_epoch=epoch)
    
    # generate 10 tracks using random seeds
    X, y = val_generator.next()
    generated = model_utils.generate_notes_from_intervals(model, X, length=200)
    for i, midi in enumerate(generated):
        file = os.path.join(model_dir, 'generated', '{}.mid'.format(i + 1))
        midi.write(file.format(i + 1))
        print('wrote midi file to {}'.format(file))
    
if __name__ == '__main__':
    main()