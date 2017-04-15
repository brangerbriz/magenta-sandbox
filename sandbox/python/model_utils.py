import os, glob, random, pdb
import numpy as np
import midi_utils
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

def get_callbacks(model_dir, checkpoint_monitor='val_acc', model_index=0):
    
    callbacks = []
    
    # save model checkpoints
    filepath = os.path.join(model_dir, 
                            'checkpoints', 
                            'model-' + 
                             str(model_index) + 
                             '_checkpoint-epoch_{epoch:03d}-val_acc_{val_acc:.3f}.hdf5')

    callbacks.append(ModelCheckpoint(filepath, 
                                     monitor=checkpoint_monitor, 
                                     verbose=1, 
                                     save_best_only=False, 
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

def load_model_from_checkpoint(model_dir, model_index=0):

    '''Loads the best performing model from checkpoint_dir'''
    with open(os.path.join(model_dir, 'model_{}.json'.format(model_index)), 'r') as f:
        model = model_from_json(f.read())

    epoch = 0
    newest_checkpoint = max(glob.iglob(model_dir + '/checkpoints/model-{}_*.hdf5'.format(model_index)), 
                            key=os.path.getctime)

    if newest_checkpoint: 
       epoch = int(newest_checkpoint[-22:-19])
       model.load_weights(newest_checkpoint)

    return model, epoch

def save_model(model, model_dir, model_index=0):
    with open(os.path.join(model_dir, 'model_{}.json'.format(model_index)), 'w') as f:
        f.write(model.to_json())


# if the model dir doesn't exist
# create it and its subfolders
def create_model_dir(model_dir):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        os.mkdir(os.path.join(model_dir, 'checkpoints'))
        os.mkdir(os.path.join(model_dir, 'generated'))

def generate_notes(model, seeds, window_size=20, length=1000, num_to_gen=10):
    
    # generate a pretty midi file from a model using a seed
    def _gen(model, seed, window_size, length):
        
        generated = []
        # ring buffer
        buf = np.copy(seed).tolist()
        while len(generated) < length:
            arr = np.expand_dims(np.asarray(buf), 0)
            pred = model.predict(arr)
            
            # argmax sampling (NOT RECOMMENDED), or...
            # index = np.argmax(pred)
            
            # prob distrobuition sampling
            index = np.random.choice(range(0, seed.shape[1]), p=pred[0])
            pred = np.zeros(seed.shape[1])

            pred[index] = 1
            generated.append(pred)
            buf.pop(0)
            buf.append(pred)

        return generated

    midis = []
    for i in range(0, num_to_gen):
        seed = seeds[random.randint(0, len(seeds) - 1)]
        gen = _gen(model, seed, window_size, length)
        midis.append(midi_utils.network_output_to_midi(gen))
    return midis

# alias
generate = generate_notes

def generate_notes_from_intervals(model, 
                                  seeds,
                                  start_note=60, 
                                  window_size=20, 
                                  length=1000, 
                                  num_to_gen=10):

    # generate a pretty midi file from a model using a seed
    def _gen(model, seed, window_size, length):
        
        generated = []
        # ring buffer
        buf = np.copy(seed).tolist()
        while len(generated) < length:
            arr = np.expand_dims(np.asarray(buf), 0)
            pred = model.predict(arr)
            
            # argmax sampling (NOT RECOMMENDED), or...
            # index = np.argmax(pred)
            
            # prob distrobuition sampling
            index = np.random.choice(range(0, seed.shape[1]), p=pred[0])
            pred = np.zeros(seed.shape[1])

            pred[index] = 1
            generated.append(pred)
            buf.pop(0)
            buf.append(pred)

        return generated

    midis = []
    for i in range(0, num_to_gen):
        seed = seeds[random.randint(0, len(seeds) - 1)]
        gen = _gen(model, seed, window_size, length)
        midis.append(midi_utils.network_output_intervals_to_midi(gen, start_note))
    return midis

def generate_timings(model, seeds, window_size=20, length=1000, num_to_gen=10):
        
    # generate a pretty midi file from a model using a seed
    def _gen(model, seed, window_size, length):
        
        generated = []
        # ring buffer
        buf = np.copy(seed).tolist()
        while len(generated) < length:
            arr = np.expand_dims(np.asarray(buf), 0)
            pred = model.predict(arr)[0]
            
            # # argmax sampling (NOT RECOMMENDED), or...
            # # index = np.argmax(pred)
            
            # # prob distrobuition sampling
            # index = np.random.choice(range(0, seed.shape[1]), p=pred[0])
            # pred = np.zeros(seed.shape[1])

            # pred[index] = 1
            generated.append(pred)
            buf.pop(0)
            buf.append(pred)

        return generated

    midis = []
    for i in range(0, num_to_gen):
        seed = seeds[random.randint(0, len(seeds) - 1)]
        gen = _gen(model, seed, window_size, length)
        midis.append(gen)
    return midis
