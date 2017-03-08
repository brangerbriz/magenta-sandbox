import dill as pickle
import numpy as np
import time, numbers, os, pdb, json, operator
import shutil
from music21 import *

with open(os.path.expanduser('~') + '/Documents/code/midi-dataset/data/match_scores.json', 'r') as f:
    MATCH_SCORES = json.load(f)

def get_midi_to_track_lut():
    mid_to_track = {}
    for k, vals in MATCH_SCORES.items():
        for v in vals:
            if not v in mid_to_track:
                mid_to_track[v] = {
                    'track': k,
                    'confidence': MATCH_SCORES[k][v]
                }
            # if the new confidence score is higher than the last
            # update the track it points to
            elif MATCH_SCORES[k][v] > mid_to_track[v]['confidence']:
                mid_to_track[v]['track'] = k
    return mid_to_track

# returns the average midi track msd alignment 
# confidence score of all files in lakh midi dataset
def get_avg_msd_midi_alignment_conf():
	mid_to_track = get_midi_to_track_lut()
	s = 0
	for k, v in mid_to_track.items():
		s += mid_to_track[k]['confidence']
	return s / len(list(mid_to_track.keys()))

def query_dict_array(arr, query):
    
    if type(query) != list:
        query = [query]
    
    def array_filter(data):
        for q in query:
            t = 'OR'
            if 'type' in q:
                if q['type'].upper() == 'AND':
                    t = 'AND'    
            for key in q:
                
                if key == 'type':
                    continue
                
                if t == 'OR':
                    if type(q[key]) == str or isinstance(q[key], numbers.Number):
                        if type(data[key]) == str or isinstance(data[key], numbers.Number):
                            if q[key] == data[key]:
                                return True
                        elif type(data[key]) == list or type(data[key]) == np.ndarray:
                            if q[key] in data[key]:
                                return True
                    elif type (q[key]) == list:
                        for v in q[key]:
                            if type(data[key]) == str or isinstance(v, numbers.Number):
                                if v == data[key]:
                                    return True
                            elif type(data[key]) == list or type(data[key]) == np.ndarray:
                                if v in data[key]:
                                    return True
                    elif callable(q[key]):
                        if q[key](data[key]):
                            return True
                else:
                    if type(q[key]) == str or isinstance(q[key], numbers.Number):
                        if type(data[key]) == str or isinstance(data[key], numbers.Number):
                            if q[key] != data[key]:
                                return False
                        elif type(data[key]) == list or type(data[key]) == np.ndarray:
                            if q[key] not in data[key]:
                                return False
                    elif type (q[key]) == list:
                        for v in q[key]:
                            if type(data[key]) == str or isinstance(v, numbers.Number):
                                if v != data[key]:
                                    return False
                            elif type(data[key]) == list or type(data[key]) == np.ndarray:
                                if v not in data[key]:
                                    return False
                    elif callable(q[key]):
                        if not q[key](data[key]):
                            return False
            if t == 'AND':
                return True
        return False

    return filter(array_filter, arr)

def create_symlink_dir(paths, dirname, file_limit_per_track=None):
    '''Takes a list of msd paths and creates a new directory 
       dirname with symbolic links to the files pointed to by paths.'''
    
    # delete dirname if it exists
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    
    # create dirname if it doesn't exist
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        
    dupes = 0
    for path in paths:
        basename = os.path.basename(path)
        # http://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        midi = max(MATCH_SCORES[basename].iteritems(), key=operator.itemgetter(1))[0]
        filename = os.path.join(path, midi + '.mid')
        if os.path.exists(filename):
            symlink = os.path.join(dirname, midi + '.mid')
            if os.path.exists(symlink):
                dupes = dupes + 1
                continue
            os.symlink(filename, symlink)
    print('found {} duplicate midi files'.format(dupes))

# extract features from a midi file using music21 + jSymbolic
def extract_features(midi_path):

    try:
        s = converter.parse(midi_path)
    except:
        return None
    
    # for a description of features, see section 5 of the jSymbolic paper
    # http://jmir.sourceforge.net/publications/MA_Thesis_2004_Bodhidharma.pdf
    feats = [
        # instrument features-----------------------------------
#         features.jSymbolic.PitchedInstrumentsPresentFeature,
#         features.jSymbolic.NotePrevalenceOfPitchedInstrumentsFeature,
        features.jSymbolic.NumberOfPitchedInstrumentsFeature,
        features.jSymbolic.StringKeyboardFractionFeature,
        features.jSymbolic.AcousticGuitarFractionFeature,
        features.jSymbolic.ElectricGuitarFractionFeature,
        features.jSymbolic.ViolinFractionFeature,
        features.jSymbolic.SaxophoneFractionFeature,
        features.jSymbolic.BrassFractionFeature,
        features.jSymbolic.WoodwindsFractionFeature,
        features.jSymbolic.OrchestralStringsFractionFeature,
        features.jSymbolic.StringEnsembleFractionFeature,
        features.jSymbolic.ElectricInstrumentFractionFeature,
        # rythm features----------------------------------------
        features.jSymbolic.InitialTempoFeature,
        features.jSymbolic.InitialTimeSignatureFeature,
        features.jSymbolic.NoteDensityFeature,
        features.jSymbolic.AverageNoteDurationFeature,
        features.jSymbolic.AverageTimeBetweenAttacksFeature,
        features.jSymbolic.VariabilityOfTimeBetweenAttacksFeature,
        features.jSymbolic.AverageTimeBetweenAttacksForEachVoiceFeature,
        features.jSymbolic.AverageVariabilityOfTimeBetweenAttacksForEachVoiceFeature,
        features.jSymbolic.ChangesOfMeterFeature,
        features.jSymbolic.MaximumNoteDurationFeature,
        features.jSymbolic.MinimumNoteDurationFeature,
        # pitch statistics features-----------------------------
        features.jSymbolic.MostCommonPitchPrevalenceFeature,
        features.jSymbolic.MostCommonPitchClassPrevalenceFeature,
        features.jSymbolic.PitchVarietyFeature,
        features.jSymbolic.PitchClassVarietyFeature,
        features.jSymbolic.RangeFeature,
        features.jSymbolic.ImportanceOfBassRegisterFeature,
        features.jSymbolic.ImportanceOfMiddleRegisterFeature,
        features.jSymbolic.ImportanceOfHighRegisterFeature,
        features.jSymbolic.MelodicIntervalHistogramFeature,
        features.jSymbolic.BasicPitchHistogramFeature,
        features.jSymbolic.PitchClassDistributionFeature,
        features.jSymbolic.QualityFeature, # mode
#         features.jSymbolic.GlissandoPrevalenceFeature,
#         features.jSymbolic.VibratoPrevalenceFeature,
        # melody features---------------------------------------
        features.jSymbolic.MelodicIntervalHistogramFeature,
        features.jSymbolic.AmountOfArpeggiationFeature,
        features.jSymbolic.RepeatedNotesFeature,
        features.jSymbolic.ChromaticMotionFeature,
        features.jSymbolic.StepwiseMotionFeature,
        features.jSymbolic.MelodicThirdsFeature,
        features.jSymbolic.MelodicFifthsFeature,
        features.jSymbolic.MelodicTritonesFeature,
        features.jSymbolic.MelodicOctavesFeature,
        features.jSymbolic.DirectionOfMotionFeature
    ]
    
    ds = features.DataSet(classLabel=midi_path)
    ds.addData(s)
    ds.addFeatureExtractors(feats)
    ds.process()
    return (ds.getFeaturesAsList(), ds.getAttributeLabels())