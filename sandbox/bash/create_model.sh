EXPERIMENT_NAME="02_basic_rnn_jazz"
MODEL_TYPE='basic_rnn'

MODEL_DIRECTORY="../../models/custom"
EXPERIMENT_DIR="$MODEL_DIRECTORY/$EXPERIMENT_NAME"

LMD_PATH='/home/bbpwn2/Documents/code/midi-dataset/data/lmd_matched'
MSD_CACHE_PICKLE='../../data/msd.pickle'

# create the experiment directory
mkdir -p $EXPERIMENT_DIR/sequence_examples
mkdir -p $EXPERIMENT_DIR/run_dir
mkdir -p $EXPERIMENT_DIR/data

# add this script to the experiment directory for reference
echo "Copying files to $EXPERIMENT_DIR..."
cat ./create_model.sh > $EXPERIMENT_DIR/create_model.sh
cat "../python/query_and_symlink.py" > "$EXPERIMENT_DIR/query_and_symlink.py"

echo "Querying datset and creating symlink directory..."
python "../python/query_and_symlink.py" \
	--lmd_matched $LMD_PATH \
	--symlink_dir $EXPERIMENT_DIR/data \
	--msd_cache $MSD_CACHE_PICKLE

# # TFRecord file that will contain NoteSequence protocol buffers.
# SEQUENCES_TFRECORD="$EXPERIMENT_DIR/notesequences.tfrecord"

# echo "Creating note sequences tfrecord..."
# convert_dir_to_note_sequences \
#   --input_dir=$EXPERIMENT_DIR/data \
#   --output_file=$SEQUENCES_TFRECORD \
#   --recursive 

# echo "Creating sequence examples dataset..."
# melody_rnn_create_dataset \
# --config=$MODEL_TYPE \
# --input=$SEQUENCES_TFRECORD \
# --output_dir=$EXPERIMENT_DIR/sequence_examples \
# --eval_ratio=0.30

# echo "Training model..."
# melody_rnn_train \
# --config=$MODEL_TYPE \
# --run_dir=$EXPERIMENT_DIR/run_dir/ \
# --sequence_example_file=$EXPERIMENT_DIR/sequence_examples/training_melodies.tfrecord \
# --hparams="{'batch_size':64,'rnn_layer_sizes':[64,64]}" \
# --num_training_steps=100000

# Todo: can evaluate model at same time in parallel here.

# echo "Generating melodies using trained model..."
# melody_rnn_generate \
# --config=$MODEL_TYPE \
# --run_dir=$EXPERIMENT_DIR/run_dir/train \
# --output_dir=$EXPERIMENT_DIR/generated \
# --num_outputs=10 \
# --num_steps=256 \
# --hparams="{'batch_size':64,'rnn_layer_sizes':[64,64]}" \
# --primer_melody="[60]"