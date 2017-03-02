import sys
sys.path.append('../python')

import utils
import dill as pickle
import time, argparse
from pprint import pprint

parser = argparse.ArgumentParser(description='Query lmd dataset and create'
                                             'a directory filled with symlinks to the results')
parser.add_argument('-s', '--symlink_dir', default=None)
parser.add_argument('-l', '--lmd_matched', default=None)
parser.add_argument('-c', '--msd_cache', required=True)
args = parser.parse_args()

#Load MSD Cache
start_time = time.time()
with open(args.msd_cache, 'r') as f:
    msd = pickle.load(f)
print('Loaded msd cache in {:.2f} seconds'.format(time.time() - start_time))

# Query dataset ------------------------------------------------------------
query = {
    'song_hotttnesss': lambda hotttnesss: hotttnesss > 0.6,
    'artist_terms': ['electronic'],
    'song_year': lambda year: year > 1990,
    'type': 'AND'
}

#---------------------------------------------------------------------------
print('Query dictionary:')
pprint(query)

results = utils.query_dict_array(msd, query)

print('Query returned {} results'.format(len(results)))

if args.symlink_dir is not None and args.lmd_matched is not None:

	print('lmd_matched: {}'.format(args.lmd_matched))
	print('symlink_dir: {}'.format(args.symlink_dir))

	# create simlink
	args.lmd_matched = args.lmd_matched + '/{}'	
	tracks = [args.lmd_matched.format(r['path']) for r in results]
	utils.create_symlink_dir(tracks, args.symlink_dir, file_limit_per_track=1)
