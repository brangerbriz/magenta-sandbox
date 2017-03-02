import dill as pickle
import numpy as np
import time, numbers, os, pdb
import shutil

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
        if os.path.exists(path):
            for i, filename in enumerate(os.listdir(path)):
                if file_limit_per_track != None and i == file_limit_per_track:
                    break
                if os.path.exists(os.path.join(dirname, filename)):
                    dupes = dupes + 1
                    continue
                os.symlink(os.path.join(path, filename), os.path.join(dirname, filename))
    print('found {} duplicate midi files'.format(dupes))