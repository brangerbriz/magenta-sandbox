import argparse, os
import pretty_midi

def log(message, verbose):
	if verbose:
		print(message)

def parse_midi(path):
    midi = None
    with open(path, 'r') as f:
        try:
            midi = pretty_midi.PrettyMIDI(f)
            midi.remove_invalid_notes()
        except:
            pass
    return midi

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', '-i', required=True,
						help='Input midi file or directory.')
	parser.add_argument('--output', '-o',
		                default='out.mid',
		                help='Output midi file or directory.'
		                	 ' Defaults to out.mid or out/.')
	parser.add_argument('--program', '-p', type=int, default=0,
		                help='Program number to use when replacing all'
		                      ' instruments.')
	parser.add_argument('--note', '-n', type=int, default=60,
		                help='Midi note number to replace all notes with.')
	parser.add_argument('--verbose', '-v', type=bool )
	return parser.parse_args()

def main():
	
	args = parse_args()
	if os.path.isdir(args.input):
		files = [os.path.join(args.input, f) \
			     for f in os.listdir(args.input) if '.mid' in f]
		if args.output == 'out.mid':
			args.output = 'out'
		if not os.path.isdir(args.output):
			os.mkdir(args.output)
			log('[*] Created directory {}'.format(args.output), arbs.verbose)
	elif os.path.exists(args.input):
		files = [ args.input ]
	else:
		print('Error: no such file or directory {}'.format(args.input))
		exit(1)

	for file in files:
		midi = parse_midi(file)
		if midi is not None:
			log('[*] Parsed {}'.format(file), args.verbose)
			for instrument in midi.instruments:
				instrument.program = args.program
				log('[*] Changed instrument program '
					'number to {}'.format(args.program), args.verbose)
				for note in instrument.notes:
					note.pitch = args.note
				log('[*] Changed {} notes to {}'.format(len(instrument.notes),
					                                      args.note), 
														  args.verbose)
			if len(files) == 1:
				filename = args.output
			else:
				filename = os.path.join(args.output, os.path.basename(file))
			midi.write(filename)
			log('Wrote 1 file to {}'.format(filename), args.verbose)
		else:
			log('[*] Failed to parse {}'.format(file), args.verbose)

if __name__ == '__main__':
	main()