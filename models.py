import tensorflow as tf

def sign_tf(x, threshold=0):
    '''
    Return a tensorflow operation for evaluating x >= threshold

    Arguments
    - x: tf.Tensor, shape = arbitrary
    - threshold: scalar, default = 0

    Returns
    - Tensorflor operation
    '''
    return tf.cast(tf.greater_equal(x, threshold), tf.int32)

class DNN():

    def __init__(self, num_features, node_array, kernel_reg_const, rand_seed):
        '''
        Arguments
        - num_features: int
            Number of features
        - node_array: numpy.ndarray
            1-D array of number of nodes in hidden layers
        - kernel_reg_const: float
            L2 regularization weight
        - rand_seed: int or None
            Seed to for numpy and tensorflow random number generators.
            Set to None to use random seed
        '''
        # Model parameters
        self.num_features = num_features
        self.node_array = node_array
        self.kernel_reg_const = kernel_reg_const
        self.rand_seed = rand_seed

        # Placeholders
        # - x: size = (batch_size, num_features)
        #       Input layer
        # - y_labels: size = (batch_size,)
        #       Labels
        # - q: scalar
        #       Loss function weight to account for imbalanced datasets. Relative cost of a positive
        #       error (incorrect classification of a positive data point) relative to a negative error.
        self.x = tf.placeholder(tf.float32, shape=(None, num_features), name='x')
        self.y_labels = tf.placeholder(tf.float32, name='y_labels') # domain: {0,1}
        self.q = tf.placeholder(tf.float32, name='q')

        # y values
        # - y_logit: real-valued logit of classifying an input data point into the positive class
        # - y_pred: predicted value: 0 or 1
        # - y_prob: probability of classifying into positive class
        self.y_logit = self.model()
        self.y_pred = sign_tf(self.y_logit)
        self.y_prob = self.prob()               

        # Build the graph for the deep net
        self.loss_fn = self.loss()          # must be assigned after self.y_logit
        self.train_step = self.optimizer()  # must be assigned after self.loss_fn

        # Evaluation functions
        self.acc_fn = self.accuracy()
        self.auroc_fn = self.auroc()

    def prob(self):
        '''
        Return the probability of classifying into the positive class
        '''
        return tf.sigmoid(self.y_logit, name='y_prob')

    def model(self):
        """
        Neural network model. Builds the graph for learning the logit.
        The probability of classifying into the positive class = sigmoid(logit)

        Returns
        - y: tf.Tensor. shape = (batch_size,)
            Logits of classifying an input data point into the positive class
        """        
        layers = []

        # input layer
        layers.append(self.x)
        
        # hidden layers
        num_hidden_layers = 0
        if self.node_array[0]:
            # The first element of self.node_array is not 0
            num_hidden_layers = self.node_array.size
        for i in range(num_hidden_layers):
            layer_hidden = tf.layers.dense(
                inputs=layers[i],
                units=self.node_array[i],
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_reg_const),
                kernel_initializer=tf.glorot_uniform_initializer(seed=None if self.rand_seed == None else i + self.rand_seed),
                name='dense'+str(i)
            )
            layers.append(layer_hidden)

        # output layer
        layer_out = tf.layers.dense(
            inputs=layers[num_hidden_layers],
            units=1,
            activation=None,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_reg_const),
            kernel_initializer=tf.glorot_uniform_initializer(seed=None if self.rand_seed == None else num_hidden_layers + self.rand_seed),
            name='output'
        )
        layers.append(layer_out)

        return tf.squeeze(layers[-1], name='y_logit')

    def optimizer(self):
        '''
        Returns the Adam optimizer
        '''
        return tf.train.AdamOptimizer().minimize(self.loss_fn)

    def loss(self):
        '''
        Returns the weighted sigmoid cross entropy loss function. The weight refers to the relative
        cost of a positive error (incorrect classification of a positive data point) relative to a
        negative error.
        '''
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.y_labels, logits=self.y_logit, pos_weight=self.q), name='loss')

    def accuracy(self):
        '''
        Returns the accuracy of the prediction, obtained by thresholding the logit.
        Accuracy = (TP + TN) / (TP + TN + FP + FN) = # correct / # data points
        '''
        correct_prediction = tf.equal(sign_tf(self.y_logit), tf.cast(self.y_labels, tf.int32))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(correct_prediction, name='accuracy')

    def auroc(self):
        '''
        Returns the area under the receiver operating characteristic curve.

        See how to use tf.metrics.auc here:
        https://stackoverflow.com/questions/45808121/why-my-tensorflow-auc-is-0-0
        '''
        # tensorflow method
        auc, update_op = tf.metrics.auc(self.y_labels, self.y_pred)
        return update_op