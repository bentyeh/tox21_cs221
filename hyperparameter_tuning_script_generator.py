import itertools as it
import util

# Parameters that are the same for all runs
# Exception: run_id changes for each run
params = {
    'run_id': 0,
    'rand_seed': 848,
    'data_dir': 'data_features',
    'data_file_ext': 'features',
    'results_dir': 'results',
    'results_file_ext': 'results',
    'res_freq': 50,
    'num_epochs': 4,
    'save_weights': True,
    'plot': True,
    'saliency': False
}

# Parameters that change over runs
assay_names = ['nr-ahr'] # 'nr-ar-lbd', 'nr-aromatase', 'nr-ar', 'nr-er-lbd', 'nr-er', 'nr-ppar-gamma', 'sr-are', 'sr-atad5', 'sr-hse', 'sr-mmp', 'sr-p53']
loss_balances = (True, False)
kernel_reg_consts = [0.01]
batch_sizes = [50]
node_arrays = set()

node_arrays.add('0') # no hidden layers

num_nodes = [256]
num_hidden_layers = [3]

# Generate all permutations of node arrays
for nl in num_hidden_layers:
    combinations = list(it.combinations_with_replacement(num_nodes,nl))
    for combination in combinations:
        permutations = list(it.permutations(combination))
        for node_array in permutations:
            # node_array is currently a tuple
            node_array = str(node_array)
            node_array = node_array.split(',')
            for i in range(len(node_array)):
                node_array[i] = node_array[i].strip('() ')
            node_array = ' '.join(node_array)
            node_arrays.add(node_array)

# Generate script
output_script = 'hyperparameter_tuning_script'
program = 'train.py'
python_opts = '-u'
redir_stdout = True
bg = False

with open(output_script, 'w') as f:
    for assay_name in assay_names:
        params['assay_name'] = assay_name
        for loss_balance in loss_balances:
            params['loss_balance'] = loss_balance
            for kernel_reg_const in kernel_reg_consts:
                params['kernel_reg_const'] = kernel_reg_const
                for batch_size in batch_sizes:
                    params['batch_size'] = batch_size
                    for node_array in node_arrays:
                        params['node_array'] = node_array

                        command = util.command_str(program, params, python_opts=python_opts, redir_stdout=redir_stdout, bg=bg)
                        f.write(command + '\n')
                        params['run_id'] += 1

'''
Available arguments

# logistical information
parser.add_argument('--run_id', type=str, default='', help='run id')
parser.add_argument('--rand_seed', type=int, default=None, help='graph-level random seed for tensorflow')
parser.add_argument('--assay_name', type=str, required=True, help='assay name, e.g. nr-ar, sr-are, ...')
parser.add_argument('--data_dir', type=str, default='data_features', help='directory of train, test, and score data files')
parser.add_argument('--data_file_ext', type=str, default='features', help='data file extension, exluduing the period (e.g. ''fp'', ''data'', etc)')
parser.add_argument('--results_dir', type=str, default='results', help='directory to save results (e.g. ''results'')')
parser.add_argument('--results_file_ext', type=str, default='results', help='results file extension, exluduing the period (e.g. ''fp'', ''data'', etc)')

# evaluation and output parameters
parser.add_argument('--res_freq', type=int, default=25, help='how often (per number of SGD batches) to save model evaluation results while training, default = 25')
parser.add_argument('--save_weights', type=util.str2bool, default=True, help='save the tensorflow model weights, default = True')
parser.add_argument('--plot', type=util.str2bool, default=False, help='show and save timeplots of accuracy metrics, default = False')
parser.add_argument('--saliency', type=util.str2bool, default=False, help='compute saliency map, default = False. requires save_weights to be True')

# training parameters
parser.add_argument('--loss_balance', type=util.str2bool, default=False, help='adjust loss function to account for unbalanced dataset, default = False')
parser.add_argument('--kernel_reg_const', type=float, default=0.01, help='L2 kernel regularization constant')
parser.add_argument('--batch_size', type=int, default=1, help='batch size. default = 1 (SGD)')
parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs (passes through entire training set)')

# model parameters
parser.add_argument('--node_array', type=int, nargs='*', required=True, help='sizes of hidden layers in the neural network. use 0 for a simple linear classifier')
'''