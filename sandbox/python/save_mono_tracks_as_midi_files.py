import os, time, uuid
import midi_utils
import pretty_midi
from multiprocessing import Pool as ThreadPool

OUTPUT_DIR='../../data/lmd_mono_tracks_seperated'

def process_midi(path):
    midi = midi_utils.parse_midi(path)
    if midi is not None:
	    mono_instruments = midi_utils.filter_monophonic(midi.instruments, 1.0)
	    for instrument in mono_instruments:
	    	if len(instrument.notes) > 10:
		    	first_note_offset = instrument.notes[0].start
		    	for note in instrument.notes:
		    		note.start = note.start - first_note_offset
		    		note.end = note.end - first_note_offset
		    	new_midi = pretty_midi.PrettyMIDI()
		    	new_midi.instruments.append(instrument)
		    	filename = '{}_{}.mid'.format(os.path.basename(path), uuid.uuid4())
		    	new_midi.write(os.path.join(OUTPUT_DIR, filename))
		    	print('{}'.format(filename))

def main():

	flat_lmd_dir = os.path.expanduser('~') + '/Documents/code/midi-dataset/data/lmd_full_flat_symlink'
	files = [os.path.join(flat_lmd_dir, name) for name in os.listdir(flat_lmd_dir)]

	num_threads = 6
	pool = ThreadPool(num_threads)

	start_time = time.time()
	pool.map(process_midi, files)
	print('Finished in {:.2f} seconds'.format(time.time() - start_time))

if __name__ == '__main__':
	main()
