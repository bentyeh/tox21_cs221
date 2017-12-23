import os
import numpy as np

def sign(x, threshold=0):
    '''
    Return True if x >= threshold

    Arguments
    - x: numpy.ndarray
    - threshold: scalar, default = 0

    Returns
    - y: numpy.ndarray, dtype = int
        y[i] = 1 if x[i] >= threshold, 0 otherwise
    '''
    y = x >= threshold
    return y.astype(int)

def get_data_filenames(data_dir, data_file_ext, assay_name):
    '''
    Returns dictionary mapping datasets to corresponding filenames

    Arguments
    - data_dir: str
        Directory of train, test, and score data files
    - data_file_ext: str
        Data file extension, exluduing the period. Example: 'features'
    - assay_name: str
        Assay name. Example: 'nr-ahr'

    Returns
    - filenames: dict, str (dataset) -> str (filenames)
        Dictionary mapping 'train', 'test', and 'score' to the corresponding data filename
        Example: filenames['train']
    '''
    return {subfolder: os.path.join('.', data_dir, subfolder, '') +
        assay_name + '.' + data_file_ext for subfolder in ['train', 'test', 'score']}

def read_features(filename, header=None, sep='\t'):
    '''
    Read features data file

    Arguments
    - filename: str
        Data must be arranged as follows:
        cid, ncats_id, smiles code, label, fingerprint, extra features (sep-delimited, same number of features per data point)
    - header: int, default = None
        Number of lines to ignore.
        Ex: header = None will read the first row as data.
        Ex: header = 0 will skip the first row (0-indexed) and begin reading data from the second row.
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
        if header != None:
            for _ in range(header+1):
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
    Convert string to boolean

    Arguments
    - v: str
        String representing boolean value, case agnostic
        Accepted values:
            True: {'yes', 'true', 't', 'y', '1'}
            False: {'no', 'false', 'f', 'n', '0'}

    Returns
    - Boolean
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

def command_str(program, params, python_opts='', redir_stdout=False, bg=False):
    '''
    Return command to run program with specified options and parameters

    Arguments
    - program: str
        Name of program to run
    - params: dict, str -> various
        Parameters to pass into the program
    - python_opts: str, default = ''
        Python arguments. See https://docs.python.org/3/using/cmdline.html
        Example: '-u' to force stdout and stderr streams to be unbuffered
        Example: '-m pdb' to run python debugger (pdb)
    - redir_stdout: bool, default = False
        Whether to redirect stdout to a file named [run_id].out
    - bg: bool, default = False
        Whether to append the bash command ' &' so that the program is run in the background

    Returns
    - command: str
        The command string with specified options and parameters
    '''
    command = 'python ' + python_opts + ' ' + program
    for key in params:
        command += ' --' + key + ' ' + str(params[key])

    if redir_stdout:
        command += ' > ' + str(params['run_id']) + '.out'

    if bg:
        command += ' &'

    return command