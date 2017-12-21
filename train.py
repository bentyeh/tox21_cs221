from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, os

import sklearn as sk
from sklearn import metrics
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import util
import models

def plot_metrics(results, metrics, datasets, results_file_base):
    '''
    Plot accuracy of model classification predictions over iterations on specified datasets
    Saves figures to specified path

    Arguments
    - results: dict, str -> dict
    - metrics: list of str
    - datasets: list of str
    - results_file_base: str
    '''

    metric_names = {
        'acc': 'Accuracy',
        'loss': 'Loss',
        'auroc': 'Area Under ROC',
        'auroc_sk': 'Area Under ROC (sk)'
    }

    for metric in metrics:
        plt.figure()
        for dataset in datasets:
            plt.plot(results['steps'], results[metric][dataset], label=dataset)
        plt.xlabel('Number of SGD batches')
        plt.ylabel(metric_names[metric])
        plt.title(metric_names[metric] + ' of model predictions versus SGD iteration')
        plt.legend()
        plt.savefig(results_file_base + '_' + metric + '.png')

def plot_ROC(X, Y, model, results, datasets, results_file_base, sess=None):
    '''
    Plot receiver operating characteristic curve on specified datasets
    Saves figure to specified path

    Arguments
    - X: dict, str -> numpy.ndarray. shape = (num_examples, num_features)
        Features
    - Y: dict, str -> numpy.ndarray. shape = (num_examples,)
        Labels
    - model
    - results: dict, str -> dict
    - metrics: list of str
    - datasets: list of str
    - results_path: str
    '''

    if sess == None:
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) # necessary for auc calculation

    # KNOWN ISSUES
    # - not sure how to plot ROC corresponding to best weights - restore weights ??
    # - auroc calculated by scikit-learn seems to be different from tensorflow
    index = np.argmax(results['auroc']['test'])
    plt.figure()
    for dataset in datasets:
        # y_prob = sess.run(model.y_prob, feed_dict={model.x: X[dataset]})
        # fpr, tpr, _ = sk.metrics.roc_curve(Y[dataset], y_prob)
        fpr, tpr, _ = sk.metrics.roc_curve(Y[dataset], results['y_prob'][dataset][index])
        auroc = sk.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=dataset.title() + ' AUC = ' + str(round(auroc, 3)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend()
    plt.savefig(results_file_base + '_AUROC.png')

    index_sk = np.argmax(results['auroc_sk']['test'])
    plt.figure()
    for dataset in datasets:
        # y_prob = sess.run(model.y_prob, feed_dict={model.x: X[dataset]})
        # fpr, tpr, _ = sk.metrics.roc_curve(Y[dataset], y_prob)
        fpr, tpr, _ = sk.metrics.roc_curve(Y[dataset], results['y_prob'][dataset][index])
        auroc = sk.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=dataset.title() + ' AUC = ' + str(round(auroc, 3)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend()
    plt.savefig(results_file_base + '_AUROC2.png')

def evaluate_model(X, Y, model, q, results, datasets, sess=None):
    '''
    Arguments
    - model
    - q: float
        Loss function weight to account for imbalanced datasets. Relative cost of a positive error
        (incorrect classification of a positive data point) relative to a negative error.
    - results: dict, str -> dict
    - X: dict, str -> numpy.ndarray. shape = (num_examples, num_features)
        Features
    - Y: dict, str -> numpy.ndarray. shape = (num_examples,)
        Labels
    - datasets: list of str

    Returns
    - No return value. Appends new data to results
    '''

    if sess == None:
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) # necessary for auc calculation

    for dataset in datasets:
        loss, acc, y_prob = sess.run([model.loss_fn, model.acc_fn, model.y_prob], feed_dict={model.x: X[dataset], model.y_labels: Y[dataset], model.q: q})

        # auroc calculated by scikit-learn seems to be different from tensorflow
        auroc = sess.run(model.auroc_fn, feed_dict={model.x: X[dataset], model.y_labels: Y[dataset]})
        fpr, tpr, _ = sk.metrics.roc_curve(Y[dataset], y_prob)
        auroc_sk = sk.metrics.auc(fpr, tpr)

        results['loss'][dataset].append(loss)
        results['acc'][dataset].append(acc)
        results['auroc'][dataset].append(auroc)
        results['auroc_sk'][dataset].append(auroc_sk)
        results['y_prob'][dataset].append(y_prob)

def save_results(X, results, params, results_file_base, metrics, datasets, eval_metric, eval_dataset):
    '''
    Save parameters and model evaluation results

    Arguments
    - X: dict, str -> numpy.ndarray. shape = (num_examples, num_features)
        Features
    '''

    results_filename = results_file_base + '.' + params['results_file_ext']

    index = np.argmax(results[eval_metric][eval_dataset])
    for dataset in datasets:
        for metric in metrics:
            params[metric + '_' + dataset] = results[metric][dataset][index]
        params[dataset + '_data_size'] = X[dataset].shape[0]
    
    series = pd.Series(params)
    df = pd.DataFrame(series)
    df = df.T
    df.to_csv(results_filename, index=False)

def saliency(X, Y, model, sess=None):
    '''
    Compute the saliency map.

    Arguments
    - X: dict, str -> numpy.ndarray. shape = (num_examples, num_features)
        Features
    - Y: dict, str -> numpy.ndarray. shape = (num_examples,)
        Labels

    Source: https://azure.github.io/learnAnalytics-DeepLearning-Azure/saliency-maps.html
    '''

    if sess == None:
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) # necessary for auc calculation

    # Compute the score of the correct class for each example.
    # This gives a Tensor with shape [N], the number of examples.
    correct_scores = model.y_labels*model.y_prob + (1-model.y_labels)*(1-model.y_prob)

    # Gradient of the scores with respect to the input features x
    grads_fun = tf.gradients(correct_scores, model.x)[0]

    # Final saliency map has shape (size_training_data, num_features)
    saliency_vecs = sess.run(grads_fun, feed_dict={model.x: X['train'], model.y_labels: Y['train']})

    return saliency_vecs

def plot_saliency(saliency_vecs, num_features, results_file_base, n_top=10, n_bottom=10):
    # bar plot (mean + sample standard deviation) of saliency of all features
    # see here for meaning of features: ftp://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt
    mean_saliency = np.mean(saliency_vecs, axis=0)
    stddev_saliency = np.std(saliency_vecs, axis=0, ddof=1)
    
    plt.figure(figsize=(100,10))
    plt.bar(range(num_features), mean_saliency, width=1, yerr=stddev_saliency)
    plt.savefig(results_file_base + '_mean_saliency_all.png')

    # bar plot (mean + sample standard deviation) of saliency of top n features
    mean_saliency = np.mean(saliency_vecs, axis=0)
    stddev_saliency = np.std(saliency_vecs, axis=0, ddof=1)

    idx_sort = np.argsort(mean_saliency)

    top_ind = idx_sort[-n_top:][::-1]
    top_val = mean_saliency[top_ind]
    top_std = stddev_saliency[top_ind]

    plt.figure() # figsize=(10,10)
    plt.bar(range(n_top), top_val, yerr=top_std, tick_label=top_ind) # width=1
    plt.xlabel('feature index') # fontsize='18'
    plt.ylabel('gradient of predicted probability of toxicity') # fontsize='18'
    plt.title('Top 10 predictive features for toxicity') # fontsize='24'
    plt.savefig(results_file_base + '_mean_saliency_top.png')

    bottom_ind = idx_sort[0:n_bottom]
    bottom_val = mean_saliency[bottom_ind]
    bottom_std = stddev_saliency[bottom_ind]

    plt.figure() # figsize=(10,10)
    plt.bar(range(n_bottom), bottom_val, yerr=bottom_std, tick_label=bottom_ind) # width=1
    plt.xlabel('feature index') 
    plt.ylabel('gradient of predicted probability of toxicity') # fontsize='18'
    plt.title('Top 10 predictive features for non-toxicity') # fontsize='24'
    plt.savefig(results_file_base + '_mean_saliency_bottom.png')

def get_data(params):
    '''
    Get data (features and labels) based on the specified parameters

    Arguments
    - params: dict, str -> str
        Parameters

    Returns
    - X: dict, str -> numpy.ndarray. shape = (num_examples, num_features)
        Features
    - Y: dict, str -> numpy.ndarray. shape = (num_examples,)
        Labels
    '''
    X, Y = {}, {}
    filenames = util.get_data_filenames(params['data_dir'], params['data_file_ext'], params['assay_name'])
    X['train'], Y['train'] = util.read_features(filenames['train'], header=1)
    X['test'], Y['test'] = util.read_features(filenames['test'], header=1)
    X['score'], Y['score'] = util.read_features(filenames['score'], header=1)
    return X, Y

def train(X, Y, params, results_file_base, eval_metric, eval_dataset):
    '''
    Arguments
    - X: dict, str -> numpy.ndarray. shape = (num_examples, num_features)
        Features
    - Y: dict, str -> numpy.ndarray. shape = (num_examples,)
        Labels
    - params: dict

    Returns
    - model
    - sess: tf.Session
    - results: dict
    '''
    # unpack params
    rand_seed = params['rand_seed']
    res_freq = params['res_freq']
    save_weights = params['save_weights']
    loss_balance = params['loss_balance']
    kernel_reg_const = params['kernel_reg_const']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    node_array = np.array(params['node_array'], dtype=int)
    num_features = params['num_features']

    # build model
    model = models.DNN(num_features, node_array, kernel_reg_const, rand_seed)

    # initialize tensorflow graph
    sess = tf.InteractiveSession()
    np.random.seed(rand_seed)
    tf.set_random_seed(rand_seed)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer()) # necessary for auc calculation: https://stackoverflow.com/questions/44422508/tensorflow-attempting-to-use-uninitialized-value-auc-auc-auc-false-positives

    # calculate frequencies of positives, negatives in training set
    q = 1
    if loss_balance:
        q = Y['train'].shape[0] / np.sum(Y['train'])
    print('q: %0.3g' % q)

    # keep track of loss and accuracy
    steps = []
    loss, acc, auroc, auroc_sk, y_prob = {}, {}, {}, {}, {}
    for res in [loss, acc, auroc, auroc_sk, y_prob]:
        res['train'] = []
        res['test'] = []

    results = {
        'steps': steps,
        'loss': loss,
        'acc': acc,
        'auroc': auroc,
        'auroc_sk': auroc_sk,
        'y_prob': y_prob
    }
    
    datasets = ['train','test']

    # evaluate model based on initialized weights
    step = 0
    steps.append(step)
    evaluate_model(X, Y, model, q, results, datasets, sess)
    
    if save_weights:
        weights_filename = results_file_base + '_model_weights.ckpt'
        saver = tf.train.Saver()
        saver.save(sess, weights_filename)

    print('(loss, acc, auroc, auroc_sk) - step %d,\t train: (%0.3g, %0.3g, %0.3g, %0.3g),\t test: (%0.3g, %0.3g, %0.3g, %0.3g)' % (
        step,
        results['loss']['train'][-1], results['acc']['train'][-1], results['auroc']['train'][-1], results['auroc_sk']['train'][-1],
        results['loss']['test'][-1], results['acc']['test'][-1], results['auroc']['test'][-1], results['auroc_sk']['test'][-1]
        )
    )

    # training loop
    num_examples = X['train'].shape[0]
    num_batches_per_epoch = int(np.ceil(num_examples / batch_size))
    print("Number of batches per epoch: %d " % num_batches_per_epoch)
    
    for epoch in range(num_epochs):
        # shuffle indices of training data
        shuffle_indices = np.arange(num_examples)
        np.random.shuffle(shuffle_indices)

        for i in range(num_batches_per_epoch):
            # get batch
            batch_indices = shuffle_indices[i*batch_size : (i+1)*batch_size]
            batch_x = X['train'][batch_indices]
            batch_y = Y['train'][batch_indices]

            # train on batch data
            sess.run(model.train_step, feed_dict={model.x: batch_x, model.y_labels: batch_y, model.q: q})
            sess.run(model.auroc_fn, feed_dict={model.x: batch_x, model.y_labels: batch_y})
            step += 1

            # store loss and accuracy
            if step % res_freq == 0 or epoch == num_epochs-1 and i == num_batches_per_epoch-1:
                evaluate_model(X, Y, model, q, results, datasets, sess)
                steps.append(step)
                print('(loss, acc, auroc, auroc_sk) - step %d,\t train: (%0.3g, %0.3g, %0.3g, %0.3g),\t test: (%0.3g, %0.3g, %0.3g, %0.3g)' % (
                    step,
                    results['loss']['train'][-1], results['acc']['train'][-1], results['auroc']['train'][-1], results['auroc_sk']['train'][-1],
                    results['loss']['test'][-1], results['acc']['test'][-1], results['auroc']['test'][-1], results['auroc_sk']['test'][-1]
                    )
                )

                # save variables only if test auroc has increased
                if save_weights:
                    if results[eval_metric][eval_dataset][-1] > max(results[eval_metric][eval_dataset]):
                        saver.save(sess, weights_filename)

    print(
        'Best ' + eval_dataset + ' ' + eval_metric + ': %0.3g' \
        % max(results[eval_metric][eval_dataset])
    )

    return model, sess, results

def main(params):
    '''
    Args
    - params: dict, str -> various types
        Parameters
    '''

    results_path = os.path.join(os.getcwd(), params['results_dir'], params['assay_name'], '')
    results_file_base = results_path + str(params['run_id'])
    if not os.path.exists(results_path):
        try:
            os.makedirs(results_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    metrics = ['loss','acc','auroc','auroc_sk']
    datasets = ['train','test']
    eval_metric = 'auroc'
    eval_dataset = 'test'

    X, Y = get_data(params)
    params['num_features'] = X['train'].shape[1]
    model, sess, results = train(X, Y, params, results_file_base, eval_metric, eval_dataset)
    save_results(X, results, params, results_file_base, metrics, datasets, eval_metric, eval_dataset)

    if params['plot']:
        plot_metrics(results, metrics, datasets, results_file_base)
        plot_ROC(X, Y, model, results, datasets, results_file_base, sess)

    saliency_vecs = None
    if params['saliency']:
        saliency_vecs = saliency(X, Y, model, sess)
        plot_saliency(saliency_vecs, params['num_features'], results_file_base)

    return model, sess, results, saliency_vecs

if __name__ == '__main__':

    ## -- construct parser -- ##
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id',	type=str, default='', help='run id')
    parser.add_argument('--rand_seed', type=int, default=None, help='graph-level random seed for tensorflow')
    parser.add_argument('--assay_name', type=str, required=True, help='assay name, e.g. nr-ar, sr-are, ...')
    parser.add_argument('--data_dir', type=str, required=True, help='directory to find train, test, and score data files')
    parser.add_argument('--data_file_ext', type=str, default='data', help='data file extension, exluduing the period (e.g. ''fp'', ''data'', etc)')
    parser.add_argument('--results_dir', type=str, default='results', help='directory to save results (e.g. ''results'')')
    parser.add_argument('--results_file_ext', type=str, default='results', help='results file extension, exluduing the period (e.g. ''fp'', ''data'', etc)')
    # output options
    parser.add_argument('--res_freq', type=int, default=25, help='how often (per number of SGD batches) to save model evaluation results while training, default = 25')
    parser.add_argument('--save_weights', type=util.str2bool, default=False, help='save the tensorflow model weights, default = False')
    parser.add_argument('--plot', type=util.str2bool, default=False, help='show and save timeplots of accuracy metrics, default = False')
    parser.add_argument('--saliency', type=util.str2bool, default=True, help='compute saliency map, default = False')
    # hyperparameters
    parser.add_argument('--loss_balance', type=util.str2bool, default=False, help='adjust loss function to account for unbalanced dataset, default = False')
    parser.add_argument('--kernel_reg_const', type=float, default=0.01, help='L2 kernel regularization constant')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default = 1 (SGD)')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs (passes through entire training set)')
    parser.add_argument('--node_array', nargs='*', required=True, help='sizes of hidden layers in the neural network. use 0 for a simple linear classifier')

    # parse arguments, convert into dictionary
    args = parser.parse_args()
    params = vars(args)

    main(params)