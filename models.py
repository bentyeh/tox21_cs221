import tensorflow as tf

def sign_tf(x, threshold=0):
    return tf.cast(tf.greater_equal(x, threshold), tf.int32)

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

class DNN():

    def __init__(self, num_features, node_array, kernel_reg_const, rand_seed):
        self.num_features = num_features
        self.node_array = node_array
        self.kernel_reg_const = kernel_reg_const
        self.rand_seed = rand_seed

        # define placeholders
        # x: input
        # y_labels: labels
        # q: loss weights for unbalanced data
        self.x = tf.placeholder(tf.float32, shape=(None, num_features))
        self.y_labels = tf.placeholder(tf.float32) # domain: {0,1}
        self.q = tf.placeholder(tf.float32)

        # Build the graph for the deep net
        self.y_logit = self.model()
        self.loss_fn = self.loss()
        self.train_step = self.optimizer()

        # Other y values
        self.y_prob = self.prob()               # probability of classifying into positive class
        self.y_pred = sign_tf(self.y_logit)     # predicted value: 0 or 1

        # Evaluation functions
        self.acc_fn = self.accuracy()
        self.auroc_fn = self.auroc()

    def prob(self):
        '''
        Return the probability of classifying into the positive class
        '''
        return tf.sigmoid(self.y_logit)

    def model(self):
        """
        Neural network model. Builds the graph for learning the logit.
        The probability of classifying into the positive class = sigmoid(logit)

        Args:
            x: tf.Tensor. size = (batch_size, num_features)
                Input layer
            nodes: numpy.ndarray
                A list of number of nodes in hidden layers
            kernel_reg_const: float
                L2 regularization weight

        Returns:
            y: a tensor of length batch_size with values equal to the logits
                of classifying an input data point into the positive class
        """        
        layers = []

        # input layer
        layers.append(self.x)
        
        # hidden layers
        num_hidden_layers = min(self.node_array.size, self.node_array[0])
        for i in range(num_hidden_layers):
            layer_hidden = tf.layers.dense(
                inputs=layers[i],
                units=self.node_array[i],
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_reg_const),
                kernel_initializer=tf.glorot_uniform_initializer(seed=None if self.rand_seed == None else i + self.rand_seed)
            )
            layers.append(layer_hidden)

        # output layer
        layer_out = tf.layers.dense(
            inputs=layers[num_hidden_layers],
            units=1,
            activation=None,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_reg_const),
            kernel_initializer=tf.glorot_uniform_initializer(seed=None if self.rand_seed == None else num_hidden_layers + self.rand_seed)
        )
        layers.append(layer_out)

        return tf.squeeze(layers[-1])

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
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.y_labels, logits=self.y_logit, pos_weight=self.q))

    def accuracy(self):
        '''
        Returns the accuracy of the prediction, obtained by thresholding the logit.
        Accuracy = (TP + TN) / (TP + TN + FP + FN) = # correct / # data points
        '''
        correct_prediction = tf.equal(sign_tf(self.y_logit), tf.cast(self.y_labels, tf.int32))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(correct_prediction)

    def auroc(self):
        '''
        Returns the area under the receiver operating characteristic curve.

        See how to use tf.metrics.auc here:
        https://stackoverflow.com/questions/45808121/why-my-tensorflow-auc-is-0-0
        '''
        auc, update_op = tf.metrics.auc(self.y_labels, self.y_pred)
        return update_op # [auc, update_op]