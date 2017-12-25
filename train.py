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
    Plot evaluation metrics of model classification predictions over iterations on specified datasets.
    Saves figures to specified path.

    Arguments
    - results: dict, str (metric) -> dict (dataset) -> list of float (evaluation results)
        Results of evaluation metrics of model classification predictions over iterations.
        Example: results[metric][dataset]
    - metrics: list of str
        Examples: 'acc', 'loss', 'auroc'
    - datasets: list of str
        Examples: 'train', 'test', 'score'
    - results_file_base: str
        Base results file name. Usually includes run_id but leaves out file extension
    '''

    # map condensed metric names to full metric names to be used as figure titles
    metric_names = {
        'acc': 'Accuracy',
        'loss': 'Loss',
        'auroc': 'Area Under ROC',
        'auroc_sk': 'Area Under ROC (sk)'
    }

    # metrics that have a range between 0 and 1
    zero_one_metrics = ['acc', 'auroc', 'auroc_sk']

    for metric in metrics:
        plt.figure()
        for dataset in datasets:
            plt.plot(results['steps'], results[metric][dataset], label=dataset)
        
        # set x- and y-axis limits
        if metric in zero_one_metrics:
            plt.ylim(0, 1)
        plt.xlim(xmin=0)

        # label axes and figure
        plt.xlabel('Number of SGD batches')
        plt.ylabel(metric_names[metric])
        plt.title(metric_names[metric] + ' of model predictions versus SGD iteration')
        plt.legend()
        plt.savefig(results_file_base + '_' + metric + '.png')

def plot_ROC(X, Y, results, datasets, results_file_base):
    '''
    Plot receiver operating characteristic curve on specified datasets.
    Save figure to specified path.

    Arguments
    - X: dict, str (dataset) -> numpy.ndarray (data). shape = (num_examples, num_features)
        Features. Example: X['train']
    - Y: dict, str (dataset) -> numpy.ndarray (data). shape = (num_examples,)
        Labels. Example: Y['train']
    - results: dict, str (metric) -> dict (dataset) -> list of float (evaluation results)
        Results of evaluation metrics of model classification predictions over iterations.
        Example: results[metric][dataset]
    - metrics: list of str
        Examples: 'acc', 'loss', 'auroc'
    - datasets: list of str
        Examples: 'train', 'test', 'score'
    - results_file_base: str
        Base results file name. Usually includes run_id but leaves out file extension
    '''

    plt.figure()
    for dataset in datasets:
        fpr, tpr, _ = sk.metrics.roc_curve(Y[dataset], results['y_prob'][dataset])
        auroc = sk.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=dataset.title() + ' AUC = ' + str(round(auroc, 3)))
    
    # set x- and y-axis limits
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    # label axes and figure
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend()
    plt.savefig(results_file_base + '_ROC.png')

def evaluate_model(X, Y, model, q, results, datasets, sess):
    '''
    Appends new evaluation values to results dictionary

    Arguments
    - X: dict, str (dataset) -> numpy.ndarray (data). shape = (num_examples, num_features)
        Features. Example: X['train']
    - Y: dict, str (dataset) -> numpy.ndarray (data). shape = (num_examples,)
        Labels. Example: Y['train']
    - model
        Prediction model
    - q: float
        Loss function weight to account for imbalanced datasets. Relative cost of a positive error
        (incorrect classification of a positive data point) relative to a negative error.
    - results: dict, str (metric) -> dict (dataset) -> list of float (evaluation results)
        Results of evaluation metrics of model classification predictions over iterations.
        Example: results[metric][dataset]
    - datasets: list of str
        Examples: 'train', 'test', 'score'
    '''

    for dataset in datasets:
        loss, acc = sess.run([model.loss_fn, model.acc_fn], feed_dict={model.x: X[dataset], model.y_labels: Y[dataset], model.q: q})

        ## AUROC
        
        # tensorflow method
        auroc = sess.run(model.auroc_fn, feed_dict={model.x: X[dataset], model.y_labels: Y[dataset]})
        
        # scikit-learm method
        y_prob = sess.run(model.y_prob, feed_dict={model.x: X[dataset]})
        fpr, tpr, _ = sk.metrics.roc_curve(Y[dataset], y_prob)
        auroc_sk = sk.metrics.auc(fpr, tpr)

        # append results
        results['loss'][dataset].append(loss)
        results['acc'][dataset].append(acc)
        results['auroc'][dataset].append(auroc)
        results['auroc_sk'][dataset].append(auroc_sk)

def save_results(X, params, results, metrics, datasets, results_file_base):
    '''
    Save parameters and model evaluation results

    Arguments
    - X: dict, str (dataset) -> numpy.ndarray (data). shape = (num_examples, num_features)
        Features. Example: X['train']
    - params: dict, str -> various
        Parameters for the model, training algorithm, and evaluation
    - results: dict, str (metric) -> dict (dataset) -> list of float (evaluation results)
        Results of evaluation metrics of model classification predictions over iterations.
        Example: results[metric][dataset]
    - metrics: list of str
        Examples: 'acc', 'loss', 'auroc'
    - datasets: list of str
        Examples: 'train', 'test', 'score'
    - results_file_base: str
        Base results file name. Usually includes run_id but leaves out file extension
    '''

    results_filename = results_file_base + '.' + params['results_file_ext']

    # add evaluation results to params
    for dataset in datasets:
        for metric in metrics:
            params[metric + '_' + dataset] = results[metric][dataset][results['best_index']]
        params[dataset + '_data_size'] = X[dataset].shape[0]
    
    # save results to file
    series = pd.Series(params)
    df = pd.DataFrame(series)
    df = df.T
    df.to_csv(results_filename, index=False, sep='\t')

    # save dtypes to file
    # dtypes_series = df.dtypes
    # dtypes_series.to_csv(results_filename + '_dtypes')

def saliency(X, Y, model, sess):
    '''
    Compute the saliency map.

    Arguments
    - X: dict, str (dataset) -> numpy.ndarray (data). shape = (num_examples, num_features)
        Features. Example: X['train']
    - Y: dict, str (dataset) -> numpy.ndarray (data). shape = (num_examples,)
        Labels. Example: Y['train']
    - model
        Prediction model
    - sess: tf.Session
        Tensorflow Session in which to compute the saliency map

    Returns
    - saliency_vecs: numpy.ndarray. shape = (num_examples, num_features)
        Saliency map. Gradient of the probability of correct classification of each example with
        respect to the features

    Source: https://azure.github.io/learnAnalytics-DeepLearning-Azure/saliency-maps.html
    '''

    # Compute the score of the correct class for each example.
    # This gives a Tensor with shape [N], the number of examples.
    correct_scores = model.y_labels*model.y_prob + (1-model.y_labels)*(1-model.y_prob)

    # Gradient of the scores with respect to the input features x
    grads_fun = tf.gradients(correct_scores, model.x)[0]

    # Final saliency map has shape (size_training_data, num_features)
    saliency_vecs = sess.run(grads_fun, feed_dict={model.x: X['train'], model.y_labels: Y['train']})

    return saliency_vecs

def plot_saliency(saliency_vecs, num_features, results_file_base, n_top=10, n_bottom=10):
    '''
    Plot the complete saliency map over all features.
    Plot the saliency map over the top n_top and bottom n_bottom features.

    Arguments
    - saliency_vecs: numpy.ndarray. shape = (num_examples, num_features)
        Saliency map. Gradient of the probability of correct classification of each example with
        respect to the features
    - num_features: int
        Number of features
    - results_file_base: str
        Base results file name. Usually includes run_id but leaves out file extension
    - n_top: int, default = 10
        Number of features that contribute most to the correct classification to plot
    - n_bottom: int, default = 10
        Number of features that contribute most to the incorrect classification to plot
    '''

    # bar plot (mean + sample standard deviation) of saliency of all features
    mean_saliency = np.mean(saliency_vecs, axis=0)
    stddev_saliency = np.std(saliency_vecs, axis=0, ddof=1)
    
    plt.figure(figsize=(100,10))
    plt.bar(range(num_features), mean_saliency, width=1, yerr=stddev_saliency)
    plt.savefig(results_file_base + '_mean_saliency_all.png')

    # Determine features that contribute most to the correct or incorrect classification
    idx_sort = np.argsort(mean_saliency)

    top_ind = idx_sort[-n_top:][::-1]
    top_val = mean_saliency[top_ind]
    top_std = stddev_saliency[top_ind]

    bottom_ind = idx_sort[0:n_bottom]
    bottom_val = mean_saliency[bottom_ind]
    bottom_std = stddev_saliency[bottom_ind]

    # bar plot (mean + sample standard deviation) of saliency of n_top features
    plt.figure() # figsize=(10,10)
    plt.bar(range(n_top), top_val, yerr=top_std, tick_label=top_ind) # width=1
    plt.xlabel('feature index') # fontsize='18'
    plt.ylabel('gradient of predicted probability of toxicity') # fontsize='18'
    plt.title('Top 10 predictive features for toxicity') # fontsize='24'
    plt.savefig(results_file_base + '_mean_saliency_top.png')

    # bar plot (mean + sample standard deviation) of saliency of n_bottom features
    plt.figure() # figsize=(10,10)
    plt.bar(range(n_bottom), bottom_val, yerr=bottom_std, tick_label=bottom_ind) # width=1
    plt.xlabel('feature index') 
    plt.ylabel('gradient of predicted probability of toxicity') # fontsize='18'
    plt.title('Top 10 predictive features for non-toxicity') # fontsize='24'
    plt.savefig(results_file_base + '_mean_saliency_bottom.png')

def get_data(params, datasets):
    '''
    Get data (features and labels) based on the specified parameters

    Arguments
    - params: dict, str -> various
        Parameters for the model, training algorithm, and evaluation
    - datasets: list of str
        Examples: 'train', 'test', 'score'

    Returns
    - X: dict, str (dataset) -> numpy.ndarray (data). shape = (num_examples, num_features)
        Features. Example: X['train']
    - Y: dict, str (dataset) -> numpy.ndarray (data). shape = (num_examples,)
        Labels. Example: Y['train']
    '''
    X, Y = {}, {}
    filenames = util.get_data_filenames(params['data_dir'], params['data_file_ext'], params['assay_name'])
    for dataset in datasets:
        X[dataset], Y[dataset] = util.read_features(filenames[dataset], header=0)
    return X, Y

def train(X, Y, params, results_file_base, eval_metric, eval_dataset):
    '''
    Create and train a model based on the input parameters

    Arguments
    - X: dict, str (dataset) -> numpy.ndarray (data). shape = (num_examples, num_features)
        Features. Example: X['train']
    - Y: dict, str (dataset) -> numpy.ndarray (data). shape = (num_examples,)
        Labels. Example: Y['train']
    - params: dict, str -> various
        Parameters for the model, training algorithm, and evaluation
    - results_file_base: str
        Base results file name. Usually includes run_id but leaves out file extension
    - eval_metric: str
        Metric to use for comparison between models. Example: 'auroc'
    - eval_dataset: str
        Dataset to use for comparison between models. Example: 'test'

    Returns
    - model
        Prediction model
    - results: dict, str (metric) -> dict (dataset) -> list of float (evaluation results)
        Results of evaluation metrics of model classification predictions over iterations.
        Example: results[metric][dataset]
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

    datasets = ['train','test']

    # build model
    model = models.DNN(num_features, node_array, kernel_reg_const, rand_seed)

    # initialize tensorflow graph
    with tf.Session() as sess:
        if params['tensorboard']:
            merged_summary = {}
            for dataset in datasets:
                loss = tf.summary.scalar('loss_' + dataset, model.loss_fn)
                acc = tf.summary.scalar('acc_' + dataset, model.acc_fn)
                merged_summary[dataset] = tf.summary.merge([loss, acc])

            # `sess.graph` provides access to the graph used in a `tf.Session`.
            tb_path = os.path.join('.', params['results_dir'], params['assay_name'], str(params['run_id']) + '_tb', '')
            params['tensorboard_path'] = tb_path
            writer = tf.summary.FileWriter(tb_path, sess.graph)

        np.random.seed(rand_seed)
        tf.set_random_seed(rand_seed)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) # necessary for auc calculation: https://stackoverflow.com/questions/44422508/tensorflow-attempting-to-use-uninitialized-value-auc-auc-auc-false-positives

        # calculate frequencies of positives, negatives in training set
        q = 1
        if loss_balance:
            q = Y['train'].shape[0] / np.sum(Y['train'])
        print('q: %0.3f' % q)
        params['q'] = q

        # keep track of loss and accuracy
        best_index = 0
        steps = []
        loss, acc, auroc, auroc_sk, y_prob = {}, {}, {}, {}, {}
        for res in [loss, acc, auroc, auroc_sk, y_prob]:
            res['train'] = []
            res['test'] = []

        results = {
            'best_index': best_index,   # index of best evaluation
            'steps': steps,             # steps
            'loss': loss,               # losses at each evaluated step
            'acc': acc,                 # accuracy at each evaluated step
            'auroc': auroc,             # auroc at each evaluated step
            'auroc_sk': auroc_sk,       # auroc (calculated with scikit-learn) at each evaluated step
            'y_prob': y_prob            # y_prob of best evaluation
        }
        
        datasets = ['train','test']

        # evaluate model based on initialized weights
        step = 0
        steps.append(step)
        evaluate_model(X, Y, model, q, results, datasets, sess)
        
        if params['tensorboard']:
            for dataset in datasets:
                s = sess.run(merged_summary[dataset], feed_dict={model.x: X[dataset], model.y_labels: Y[dataset], model.q: q})
                writer.add_summary(s, step)


        params['weights_filename'] = ''
        if save_weights:
            weights_filename = results_file_base + '_model_weights.ckpt'
            checkpoint_filename = str(params['run_id']) + '_checkpoint'
            params['weights_filename'] = weights_filename
            saver = tf.train.Saver()
            saver.save(sess, weights_filename, latest_filename=checkpoint_filename)

        print('(loss, acc, auroc, auroc_sk) - step %d,\ttrain: (%0.3f, %0.3f, %0.3f, %0.3f),\ttest: (%0.3f, %0.3f, %0.3f, %0.3f)' % (
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
                step += 1
                sess.run(model.train_step, feed_dict={model.x: batch_x, model.y_labels: batch_y, model.q: q})
                # sess.run(model.auroc_fn, feed_dict={model.x: batch_x, model.y_labels: batch_y})

                # store loss and accuracy
                if step % res_freq == 0 or epoch == num_epochs-1 and i == num_batches_per_epoch-1:
                    evaluate_model(X, Y, model, q, results, datasets, sess)
                    steps.append(step)
                    print('(loss, acc, auroc, auroc_sk) - step %d,\ttrain: (%0.3f, %0.3f, %0.3f, %0.3f),\ttest: (%0.3f, %0.3f, %0.3f, %0.3f)' % (
                        step,
                        results['loss']['train'][-1], results['acc']['train'][-1], results['auroc']['train'][-1], results['auroc_sk']['train'][-1],
                        results['loss']['test'][-1], results['acc']['test'][-1], results['auroc']['test'][-1], results['auroc_sk']['test'][-1]
                        )
                    )

                    if params['tensorboard']:
                        for dataset in datasets:
                            s = sess.run(merged_summary[dataset], feed_dict={model.x: X[dataset], model.y_labels: Y[dataset], model.q: q})
                            writer.add_summary(s, step)

                    # save variables only if eval_metric has improved
                    if save_weights and results[eval_metric][eval_dataset][-1] > max(results[eval_metric][eval_dataset][:-1]):
                        print('saving new weights')

                        # save index
                        results['best_index'] = len(results[eval_metric][eval_dataset]) - 1

                        # save y_prob
                        for dataset in datasets:
                            y_prob = sess.run(model.y_prob, feed_dict={model.x: X[dataset]})
                            results['y_prob'][dataset] = y_prob

                        # save weights
                        saver.save(sess, weights_filename, latest_filename=checkpoint_filename)

                    # check for convergence
                    if len(results[eval_metric][eval_dataset]) > 5:
                        if np.std(results[eval_metric][eval_dataset][-5:]) < 1e-5:
                            print('Convergence criteria reached.')
                            print('Best ' + eval_dataset + ' ' + eval_metric + ': %0.3f' \
                                % results[eval_metric][eval_dataset][results['best_index']])
                            return model, results

        if params['tensorboard']:
            writer.close()

    print('Maximum training steps reached.')
    print('Best ' + eval_dataset + ' ' + eval_metric + ': %0.3f' \
        % results[eval_metric][eval_dataset][results['best_index']])

    return model, results

def main(params):
    '''
    Create, train, and evaluate a model based on the input parameters

    Arguments
    - params: dict, str -> various
        Parameters for the model, training algorithm, and evaluation

    Returns
    - model
        Prediction model
    - results: dict, str (metric) -> dict (dataset) -> list of float (evaluation results)
        Results of evaluation metrics of model classification predictions over iterations.
        Example: results[metric][dataset]
    - saliency_vecs: numpy.ndarray. shape = (num_examples, num_features)
        Saliency map. Gradient of the probability of correct classification of each example with
        respect to the features
        Returns None if the parameters did not specify to compute the saliency map
    '''

    results_path = os.path.join('.', params['results_dir'], params['assay_name'], '')
    results_file_base = results_path + str(params['run_id'])
    if not os.path.exists(results_path):
        try:
            os.makedirs(results_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    metrics = ['loss','acc','auroc','auroc_sk']
    datasets = ['train','test']
    eval_metric = 'auroc_sk'
    eval_dataset = 'test'

    # get the data
    X, Y = get_data(params, datasets)
    params['num_features'] = X['train'].shape[1]

    # train and evaluate the model
    model, results = train(X, Y, params, results_file_base, eval_metric, eval_dataset)
    save_results(X, params, results, metrics, datasets, results_file_base)

    # plot evaluation metrics
    if params['plot']:
        plot_metrics(results, metrics, datasets, results_file_base)
        plot_ROC(X, Y, results, datasets, results_file_base)

    # compute and plot saliency map
    saliency_vecs = None
    if params['saliency']:
        if params['save_weights']:
            # restore weights
            weights_filename = params['weights_filename']
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess, weights_filename)
                saliency_vecs = saliency(X, Y, model, sess)
            plot_saliency(saliency_vecs, params['num_features'], results_file_base)
        else:
            print('Cannot compute saliency map without saved weights.')

    return model, results, saliency_vecs

if __name__ == '__main__':

    ## construct parser
    parser = argparse.ArgumentParser()
    
    # logistical information
    parser.add_argument('--run_id',	type=str, default='', help='run id')
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
    parser.add_argument('--tensorboard', type=util.str2bool, default=False, help='write out summary operations to results_dir/assay_name/run_id_tb/')

    # training parameters
    parser.add_argument('--loss_balance', type=util.str2bool, default=False, help='adjust loss function to account for unbalanced dataset, default = False')
    parser.add_argument('--kernel_reg_const', type=float, default=0.01, help='L2 kernel regularization constant')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default = 1 (SGD)')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs (passes through entire training set)')
    
    # model parameters
    parser.add_argument('--node_array', type=int, nargs='*', required=True, help='sizes of hidden layers in the neural network. use 0 for a simple linear classifier')

    ## parse arguments, convert into dictionary
    args = parser.parse_args()
    params = vars(args)

    main(params)