import util
import random

# Parameters that are the same for all runs
# Exception: run_id changes for each run
params = {
    'run_id': 0,
    'smiles_dir': 'data_raw',
    'smiles_file_ext': 'smiles',
    'features_dir': 'data_features',
    'features_file_ext': 'features',
    'attempts': 15,
    'delay': 1.0
}

# Parameters that change over runs
assay_names = ['nr-ahr', 'nr-ar-lbd', 'nr-aromatase', 'nr-ar', 'nr-er-lbd', 'nr-er', 'nr-ppar-gamma', 'sr-are', 'sr-atad5', 'sr-hse', 'sr-mmp', 'sr-p53']
datasets = ['train', 'test', 'score']

# Generate script
output_script = 'feature_retriever_script'
program = 'smiles_to_features.py'
python_opts = '-u'
redir_stdout = True
bg = True

with open(output_script, 'w') as f:
    for assay_name in assay_names:
        params['assay_name'] = assay_name
        for dataset in datasets:
            params['dataset'] = dataset

            # Optional code to vary the delay time
            if dataset == 'train':
                params['delay'] = random.uniform(2, 5)
            else:
                params['delay'] = random.uniform(4, 10)

            command = util.command_str(program, params, python_opts=python_opts, redir_stdout=redir_stdout, bg=bg)
            f.write(command + '\n')
            params['run_id'] += 1

'''
Available arguments
parser.add_argument('--assay_name', type=str, required=True, help='assay name, e.g. nr-ar, sr-are, ...')
parser.add_argument('--smiles_dir', type=str, required=True, help='directory to find train, test, and score smiles files')
parser.add_argument('--smiles_file_ext', type=str, default='smiles', help='smiles file extension, exluduing the period (e.g. ''smiles'')')
parser.add_argument('--dataset', type=str, required=True, help='which dataset to process: train, test, or score')
parser.add_argument('--features_dir', type=str, required=True, help='directory to store train, test, and score features files')
parser.add_argument('--features_file_ext', type=str, default='features', help='features file extension, exluduing the period (e.g. ''features'')')
parser.add_argument('--attempts', type=int, default=5, help='number of attempts to try to retrieve PubChem record of compound')
parser.add_argument('--delay', type=float, default=1.0, help='seconds to wait between each attempt at retrieving the PubChem record of a compound')
'''