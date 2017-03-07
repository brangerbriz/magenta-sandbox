## LMD + Magenta Experiments

This document is an outline of experiments conducted using the lmd_matched Lakh Midi Dataset to train various Magenta models. Experiment data, model checkpoints, etc... can be found in `models/custom`. Below you will find a description of each experiment.

### `00_basic_rnn_jazz_baseline`

Magenta `basic_rnn` model trained on ~4000 midi files from lmd_matched. This was the very first time going through the magenta model training process/pipeline and should represent proof of concept only. 

Using default/example hyperparameters. Two layer RNN w/ 64 units each.

### `01_basic_rnn_jazz`

First complete training experiment. Trained on ~10K files from artists with terms including 'jazz' from msd.

Using default/example hyperparameters. Two layer RNN w/ 64 units each.

### `02_basic_rnn_tracks_since_2000`

All tracks that have an msd year and that year is greater than 1999 (5144 files).

### `03_basic_rnn_jazz`

Using same dataset as `01_basic_rnn_jazz` but training w/ two RNN layers of 128 each.

### `04_lmd_matched`

31034 results in lmd datset. Found 13872 duplicate midi files. Processed 17003 inputs total. Produced 73081 outputs (monophonic melodies).