import os, glob, random
import numpy as np
import midi_utils
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

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


# if the model dir doesn't exist
# create it and its subfolders
def create_model_dir(model_dir):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        os.mkdir(os.path.join(model_dir, 'checkpoints'))
        os.mkdir(os.path.join(model_dir, 'generated'))

def generate(model, seeds, window_size=20, length=1000, num_to_gen=10):
    
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
        seed = seeds[random.randint(0, len(seeds))]
        gen = _gen(model, seed, window_size, length)
        midis.append(midi_utils.network_output_to_midi(gen))
    return midis
