import os, time, sys
sys.path.append('../../python')

import pdb, glob, random
import midi_utils
import pretty_midi
import numpy as np
from multiprocessing import Pool as ThreadPool
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

np.set_printoptions(threshold=np.nan)

OUTPUT_SIZE=128

# returns X, y data windows from all monophonic instrument
# tracks in a pretty midi file
def get_data_from_midi(midi):
    X, y = [], []
    for m in midi:
        if m is not None:
            melody_instruments = midi_utils.filter_monophonic(m.instruments, 
                                                              0.95)
            for instrument in melody_instruments:
                if len(instrument.notes) > 20:
                    windows = midi_utils.encode_sliding_window_notes(instrument, 
                                                                window_size=20)
                    for w in windows:
                        X.append(w[0])
                        y.append(w[1])
    return (np.asarray(X), np.asarray(y))

def batch_data(filenames):

    batch_size=32 # this refers to windows
    load_size=200 # this refers to files

     # load midi data
    num_threads = 8
    pool = ThreadPool(num_threads)

    load_index = 0

    while True:
        
        load_files = filenames[load_index:load_index + load_size]
        print('length of load files: {}'.format(len(load_files)))
        load_index = (load_index + load_size) % len(filenames)

        print('loading large batch: {}'.format(load_size))
        # print('Parsing midi files...')
        # start_time = time.time()
        parsed = map(midi_utils.parse_midi, load_files)
        # print('Finished in {:.2f} seconds'.format(time.time() - start_time))
        # print('parsed, now extracting data')
        data = get_data_from_midi(parsed)
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

# create or load a saved model
# returns the model and the epoch number (>1 if loaded from checkpoint)
def get_model(model_dir=None, window_size=20):
    
    epoch = 0
    
    if not model_dir:
        model = Sequential()
        model.add(LSTM(64, 
                       return_sequences=False, 
                       input_shape=(window_size, OUTPUT_SIZE)))
        model.add(Dropout(0.2))
        # model.add(LSTM(32, return_sequences=False))
        # model.add(Dropout(0.2))
        model.add(Dense(OUTPUT_SIZE))
        model.add(Activation('softmax'))
    else:
        model, epoch = load_model_from_checkpoint(model_dir)

    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model, epoch)
    return model, epoch


# return keras callbacks
def get_callbacks(model_dir):
    
    callbacks = []
    
    # save model checkpoints
    filepath = os.path.join(model_dir, 
                            'checkpoints', 
                            'checkpoint-epoch_{epoch:03d}-val_acc_{val_acc:.3f}.hdf5')

    callbacks.append(ModelCheckpoint(filepath, 
                                     monitor='val_acc', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='max'))

    # callbacks.append(EarlyStopping(monitor='val_loss', 
    #                                min_delta=0, 
    #                                patience=0, 
    #                                verbose=0, 
    #                                mode='auto'))

    callbacks.append(ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.5, 
                                       patience=3, 
                                       verbose=1, 
                                       mode='auto', 
                                       epsilon=0.0001, 
                                       cooldown=0, 
                                       min_lr=0))

    callbacks.append(TensorBoard(log_dir=os.path.join(model_dir, 'tensorboard-logs'), 
                                histogram_freq=0, 
                                write_graph=True, 
                                write_images=False))

    return callbacks

def load_model_from_checkpoint(model_dir):

    '''Loads the best performing model from checkpoint_dir'''
    with open(os.path.join(model_dir, 'model.json'), 'r') as f:
        model = model_from_json(f.read())

    epoch = 0

    newest_checkpoint = max(glob.iglob(model_dir + '/checkpoints/*.hdf5'), 
                            key=os.path.getctime)

    if newest_checkpoint: 
       epoch = int(newest_checkpoint[-22:-19])
       model.load_weights(newest_checkpoint)

    return model, epoch

def save_model(model, model_dir):
    with open(os.path.join(model_dir, 'model.json'), 'w') as f:
        f.write(model.to_json())

# generate a pretty midi file from a model using a seed
def generate(model, seed, window_size=20, length=1000):
    
    generated = []
    # ring buffer
    buf = np.copy(seed).tolist()
    while len(generated) < length:
        arr = np.expand_dims(np.asarray(buf), 0)
        pred = model.predict(arr)
        
        # argmax sampling, or...
        # index = np.argmax(pred)
        
        # prob distrobuition sampling
        index = np.random.choice(range(0, OUTPUT_SIZE), p=pred[0])
        pred = np.zeros(OUTPUT_SIZE)

        pred[index] = 1
        generated.append(pred)
        buf.pop(0)
        buf.append(pred)

    return generated

def main():

    midi_dir = '../../../data/query_symlinks'

    model_num = 11
    model_dir = '../../../models/keras/{}'.format(model_num)

    window_size = 20

    # if the model dir doesn't exist
    # create it and checkpoints/
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        os.mkdir(os.path.join(model_dir, 'checkpoints'))
        os.mkdir(os.path.join(model_dir, 'generated'))

    # # load midi data
    # num_threads = 8
    # pool = ThreadPool(num_threads)

    # print('Parsing midi files...')
    # files = os.listdir(midi_dir)[0:500]
    # start_time = time.time()
    # parsed = map(midi_utils.parse_midi, [os.path.join(midi_dir, f) for f in files])
    # print('Finished in {:.2f} seconds'.format(time.time() - start_time))

    # # midi = parsed[0]
    # # for instru in midi.instruments:
    # #     print('{}: {:.2f}'.format(instru.name, 
    # #                               midi_utils.get_percent_monophonic(instru.get_piano_roll())))

    # print('getting data...')
    # X, y = get_data(parsed)

    files = [os.path.join(midi_dir, path) for path in os.listdir(midi_dir)]
    train_generator = batch_data(files[0:1000])
    val_generator = batch_data(files[1000:1100])
    
    model, epoch = get_model(model_dir, window_size=window_size)
    print(model.summary())
    save_model(model, model_dir)
    
    callbacks = get_callbacks(model_dir)
    
    # print('fitting model...')
    # model.fit_generator(train_generator,
    #                     steps_per_epoch=10000, 
    #                     epochs=10,
    #                     validation_data=val_generator, 
    #                     validation_steps=2000,
    #                     verbose=1, 
    #                     callbacks=callbacks,
    #                     initial_epoch=epoch)
    
    X, y = val_generator.next()

    # # generate 10 tracks using random seeds
    for i in range(0, 10):
        seed = X[random.randint(0, len(X))]
        gen = generate(model, seed, window_size=window_size, length=100)
        midi = midi_utils.decode_sliding_windows(gen)
        file = os.path.join(model_dir, 'generated', '{}.mid'.format(i))
        midi.write(file.format(i))
        print('wrote midi file to {}'.format(file))
    
if __name__ == '__main__':
    main()