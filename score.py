# Python Standard Library
import os, argparse

# Packages
import numpy as np
import pandas as pd
import tensorflow as tf

# Own code
import util, models, train

def best_run_id(results_path, results_file_ext, eval_metric):
    '''
    Return the run_id and weights_filename of the best model

    Arguments
    - results_path: str
        Path to folder (e.g. results_dir\\assay_name\\) storing results files
    - results_file_ext: str
        Results file extension, exluduing the period (e.g. 'results')
    - eval_metric: str
        Metric to use for comparison between models. Example: 'auroc_test'
        Note: This is slightly different from the eval_metric used in train.py.
            Here, eval_metric refers to the name of the desired evaluation metric+dataset column in the results file.

    Returns
    - run_id: int
        run_id of best model
    - weights_filename: str
        Filename of saved Tensorflow weights of best model
    '''
    # Get list of all results files
    results_file_list = [os.path.join(results_path, '', x) for x in os.listdir(results_path) if x.endswith(params['results_file_ext'])]

    # Read in all results files into a pandas DataFrame
    results_df_list = []
    for results_file in results_file_list:
        df = pd.read_csv(results_file, header=0, float_precision='high', sep='\t')
        df.index = df['run_id']
        results_df_list.append(df)
    df = pd.concat(results_df_list, axis=0, join='outer')

    # Determine best run_id using eval_metric
    index = df[eval_metric].idxmax()
    run_id = df['run_id'][index]
    weights_filename = df['weights_filename'][index]
    return run_id, weights_filename

def eval_score(results_file_base, results_file_ext, weights_filename, saliency=False):
    '''
    Evaluate a trained model on the score dataset

    Arguments
    - results_file_base: str
        Base results file name. Usually includes run_id but leaves out file extension
    - results_file_ext: str
        Results file extension, exluduing the period (e.g. 'results')
    - weights_filename: str
        Filename of saved Tensorflow weights
    - saliency: bool, default = False
        Whether to compute and plot the saliency map
    '''
    # read results of best run
    results_file = results_file_base + '.' + results_file_ext
    results_file_dtypes = results_file + '_dtypes'
    # dtypes_series = pd.read_csv('dtype_series', header=None)
    # dtypes_series = dtypes_series.set_index(0).squeeze()
    # dtypes_dict = dtypes_series.to_dict()
    df = pd.read_csv(results_file, header=0, float_precision='high', sep='\t') # dtype=dtypes_dict
    series = df.iloc[0]
    params = series.to_dict()

    # Get data
    datasets = ['train','test','score']
    metrics = ['loss','acc','auroc','auroc_sk']
    X, Y = train.get_data(params, datasets)

    # unpack params
    rand_seed = params['rand_seed']
    kernel_reg_const = params['kernel_reg_const']
    num_features = params['num_features']
    q = params['q']

    node_array = params['node_array'].split(',')
    for i in range(len(node_array)):
        node_array[i] = int(node_array[i].strip('[] '))
    node_array = np.array(node_array)

    # rebuild model
    model = models.DNN(num_features, node_array, kernel_reg_const, rand_seed)

    # recreate results dict
    loss, acc, auroc, auroc_sk, y_prob = {}, {}, {}, {}, {}
    for res in [loss, acc, auroc, auroc_sk, y_prob]:
        for dataset in datasets:
            res[dataset] = []

    results = {
        'best_index': 0,
        'loss': loss,
        'acc': acc,
        'auroc': auroc,
        'auroc_sk': auroc_sk,
        'y_prob': y_prob
    }

    # restore graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, weights_filename)

        # evaluate model on all datasets, including score
        train.evaluate_model(X, Y, model, q, results, datasets, sess)
        for dataset in datasets:
            y_prob = sess.run(model.y_prob, feed_dict={model.x: X[dataset]})
            results['y_prob'][dataset] = y_prob

        # plot ROC curve and save results
        train.plot_ROC(X, Y, results, datasets, results_file_base)
        train.save_results(X, params, results, metrics, datasets, results_file_base)

        # compute and plot saliency map
        if saliency:
            saliency_vecs = train.saliency(X, Y, model, sess)
            train.plot_saliency(saliency_vecs, num_features, results_file_base)

def main(params):
    '''
    Evaluate a specific trained model on the score dataset, or find the optimal trained model and
    evaluate that model on the score dataset.
    
    Arguments
    - params: dict, str -> various
        Parameters specifying which models to evaluate
        - Note: If run_id is None, search for the optimal trained model in results_dir/assay_name/
          based on eval_metric
        - Note: These params are different from the params in train.py or those read in from the CSV
          files saved by train.py
    '''
    # unpack params
    assay_name = params['assay_name']
    results_dir = params['results_dir']
    results_file_ext = params['results_file_ext']
    run_id = params['run_id']
    eval_metric = params['eval_metric']
    saliency = params['saliency']

    results_path = os.path.join('.', results_dir, assay_name, '')

    # get best run (set of hyperparameters)
    if run_id == None:
        run_id, weights_filename = best_run_id(results_path, results_file_ext, eval_metric)
        print('Best run_id: %d' % run_id)
        print('Best weights_filename: %s' % weights_filename)
    else:
        results_file = results_path + str(run_id) + '.' + results_file_ext
        df = pd.read_csv(results_file, header=0, float_precision='high', sep='\t')
        weights_filename = df['weights_filename'][0]

    results_file_base = results_path + str(run_id)
    eval_score(results_file_base, results_file_ext, weights_filename, saliency)

if __name__ == '__main__':

    ## -- construct parser -- ##
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=None, help='run_id to use to evaluate score dataset. if not specified, the best run_id is chosen automatically')
    parser.add_argument('--assay_name', type=str, required=True, help='assay name, e.g. nr-ar, sr-are, ...')
    parser.add_argument('--results_dir', type=str, default='results', help='directory to save results (e.g. ''results'')')
    parser.add_argument('--results_file_ext', type=str, default='results', help='results file extension, exluduing the period (e.g. ''fp'', ''data'', etc)')
    parser.add_argument('--eval_metric', type=str, default='auroc_test', help='metric in results file to use to determine optimal run_id (default = ''auroc_test'')')
    parser.add_argument('--saliency', type=util.str2bool, default=False, help='compute saliency map, default = False')

    # parse arguments, convert into dictionary
    args = parser.parse_args()
    params = vars(args)

    main(params)