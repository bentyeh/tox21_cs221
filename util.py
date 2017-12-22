import os
import numpy as np

def sign(x, threshold=0):
    '''
    Parameters
    - x: numpy.ndarray
    - threshold: scalar, default = 0

    Returns
    - y: numpy.ndarray, dtype = int
        y[i] = 1 if x[i] > threshold, 0 otherwise
    '''
    y = x > threshold
    return y.astype(int)

def get_data_filenames(data_dir, data_file_ext, assay_name):
    '''
    Returns dictionary mapping 'train', 'test', and 'score' to the corresponding data filename
    '''
    return {subfolder: os.path.join('.', data_dir, subfolder, '') +
        assay_name + '.' + data_file_ext for subfolder in ['train', 'test', 'score']}

def read_features(filename, header=0, sep='\t'):
    '''
    Parameters
    - filename: str
        Data must be arranged as follows:
        cid, ncats_id, smiles code, label, fingerprint, extra features (sep-delimited, same number of features per data point)
    - header: int, default = 0
        Number of lines to ignore.
        Ex: header = 0 will read the first row as data.
        Ex: header = 1 will skip the first row (1-indexed) and begin reading data from the second row.
    - sep: str, default = '\t'
        Delimiter. Default is tab-delimited. Set to None to split by whitespace.

    Returns
    - (X, Y): tuple of np.arrays, dtype: float
        X is an array of features
        Y is a vector of labels
    '''

    X = []
    Y = []

    with open(filename, 'r') as data_file:
        # skip headers
        for _ in range(header):
            data_file.readline()
        
        for index, line in enumerate(data_file):
            split_line = line.strip().split(sep)
            label = int(split_line[3])
            fingerprint = [int(c) for c in split_line[4]]
            extra_features = split_line[5:] # if there are no extra features, assigns []

            # append data point to X (features) and Y (labels)
            X.append(fingerprint + extra_features) # +: list concatentation
            Y.append(label)

    return (np.array(X, dtype=float), np.array(Y, dtype=int))

def str2bool(v):
    '''
    Args
    - v: str
        String representing boolean value, case agnostic
        Accepted values:
            True: {'yes', 'true', 't', 'y', '1'}
            False: {'no', 'false', 'f', 'n', '0'}

    Returns
    - boolean corresponding to string argument
        Raises type error if string argument is not an accepted value

    Source
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    '''

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')