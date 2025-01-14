import numpy as np
import pickle
from numpy.random import default_rng
import math 

def xavier_init(size, gain = 1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative 
    log-likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """ 
        Constructor of the Sigmoid layer.
        """
        self._cache_current = None

    def forward(self, z):
        """ 
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            z {np.ndarray} -- Input linear output array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        self._cache_current = 1.0/(1+np.exp(-z)) 
        return self._cache_current

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_a):
        """
        Given `grad_a`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_a {np.ndarray} -- Gradient array of activations, has shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        sigmoid_grad = grad_a * self._cache_current * (1 - self._cache_current) 
        return sigmoid_grad

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class ReluLayer(Layer):
    """
    ReluLayer: Applies ReLU function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None

    def forward(self, z):
        """ 
        Performs forward pass through the ReLU layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            z {np.ndarray} -- Input linear output array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._cache_current = np.maximum(z, 0) 
        return self._cache_current

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_a):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_a {np.ndarray} -- Gradient array of activations, has shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        grad_z = grad_a
        grad_z[self._cache_current == 0] = 0
        return grad_z 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._W = xavier_init((n_in, n_out))
        self._b = xavier_init((1, n_out))

        self._cache_current = None
        self._grad_W_current = xavier_init((n_in, n_out))
        self._grad_b_current = xavier_init((1, n_out))

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        z = x @ self._W + np.repeat(self._b, x.shape[0], axis=0)
        self._cache_current = x
        return z 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._grad_W_current = self._cache_current.T @ grad_z
        self._grad_b_current = np.ones((1, self._cache_current.shape[0])) @ grad_z
        return grad_z @ self._W.T

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._W -= learning_rate * self._grad_W_current
        self._b -= learning_rate * self._grad_b_current

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding 
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        assert (len(neurons) == len(activations))
        self._layers = []
        for i in range(len(neurons)):
            if i == 0:
                self._layers.append(LinearLayer(input_dim, neurons[0]))
            else:
                self._layers.append(LinearLayer(neurons[i-1], neurons[i]))
            if activations[i] != "identity":
                self._layers.append(self._layer_selection(neurons[i], activations[i]))

    def _layer_selection(self, channel_num, type):
        if type == "relu": 
            return ReluLayer()
        elif type == "sigmoid":
            return SigmoidLayer()
        else:
            raise ValueError("type value must be one of 'relu', 'sigmoid', 'identity'")
            
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for layer in self._layers:
            x = layer(x)
        return x

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for layer in reversed(self._layers):
            grad_z = layer.backward(grad_z)
        return grad_z

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for layer in self._layers:
            layer.update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        if loss_fun == "mse":
            self._loss_layer = MSELossLayer()
        else:
            self._loss_layer = CrossEntropyLossLayer()
        
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns: 
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        shuffled_indices = default_rng().permutation(input_dataset.shape[0])
        shuffled_targets = target_dataset[shuffled_indices]
        shuffled_inputs = input_dataset[shuffled_indices]
        return shuffled_inputs, shuffled_targets 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for i in range(self.nb_epoch):
            if self.shuffle_flag == True:
                input_dataset, target_dataset = Trainer.shuffle(input_dataset,target_dataset)
            for j in range(math.ceil(input_dataset.shape[0] / self.batch_size)):
                if j == math.ceil(input_dataset.shape[0] / self.batch_size) - 1:
                    batchsize_target_dataset = target_dataset[j*self.batch_size:, :]
                    batchsize_input_dataset = input_dataset[j*self.batch_size:, :]
                else:
                    batchsize_target_dataset = target_dataset[j *self.batch_size : (j+1)*self.batch_size, :]
                    batchsize_input_dataset = input_dataset[j * self.batch_size : (j+1)*self.batch_size, :]
                output_before_loss = self.network(batchsize_input_dataset)
                self._loss_layer(output_before_loss, batchsize_target_dataset)
                self.network.backward(self._loss_layer.backward())
                self.network.update_params(self.learning_rate)
                
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data. Returns
        scalar value.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).
        
        Returns:
            a scalar value -- the loss
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        network_output = self.network(input_dataset)
        return self._loss_layer(network_output, target_dataset)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # self.mean = data.mean(axis = 0)
        # self.std = data.std(axis = 0)

        # set lower and upper bound of preprocessed data
        self.lb = 0
        self.ub = 1
        # get the min and max value of each column
        self.max = data.max(axis = 0)
        self.min = data.min(axis = 0)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # mean_ = np.repeat(self.mean.reshape(-1, self.mean.size), data.shape[0], axis = 0)
        # std_ = np.repeat(self.std.reshape(-1, self.std.size), data.shape[0], axis = 0)
        # max_ = np.repeat(self.max.reshape(-1, self.max.size), data.shape[0], axis = 0)
        # min_ = np.repeat(self.min.reshape(-1, self.min.size), data.shape[0], axis = 0)
        normalized_data = []
        for column in range(data.shape[1]):
            normalized_data_column = self.lb + ((data[:, column] - self.min[column])/(self.max[column] - self.min[column])) * (self.ub - self.lb)
            normalized_data.append(normalized_data_column)

        normalized_data = np.array(normalized_data)

        return normalized_data.T

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # mean_ = np.repeat(self.mean.reshape(-1, self.mean.size), data.shape[0], axis = 0)
        # std_ = np.repeat(self.std.reshape(-1, self.std.size), data.shape[0], axis = 0)
        # max_ = np.repeat(self.max.reshape(-1, self.max.size), data.shape[0], axis = 0)
        # min_ = np.repeat(self.min.reshape(-1, self.min.size), data.shape[0], axis = 0)

        reverted_data = []

        for column in range(data.shape[1]):
            reverted_data_column = self.min[column] + ((data[:, column] - self.lb) / (self.ub - self.lb)) * (self.max[column] - self.min[column])
            reverted_data.append(reverted_data_column)

        reverted_data = np.array(reverted_data)

        return reverted_data.T

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def LinearLayerTest():
    print("input with size (2,3) test layer out channel num is 5 ")
    lr = 0.1
    inputs = np.random.rand(2, 3)
    layer = LinearLayer(n_in=3, n_out=5)
    print("previous W:")
    print(layer._W)
    print("previous b:")
    print(layer._b)
    outputs = layer(inputs)
    grad_loss_wrt_inputs = layer.backward(outputs)
    print("bp gradient:")
    print(grad_loss_wrt_inputs)
    layer.update_params(lr)
    print("Updated W:")
    print(layer._W)
    print("Updated b:")
    print(layer._b)
    

def ReLULayerTest():
    layer = ReluLayer()
    input = np.array([[1,2,3], [-1,-2,-3]])
    print("input")
    print(input)
    output = layer(input)
    print("output")
    print(output)
    print("pass-in pL/pA")
    plpa = np.random.rand(2,3)
    print(plpa)
    print("bp result")
    print(layer.backward(plpa))


def SigmoidLayerTest():
    layer = SigmoidLayer()
    input = np.array([[1,0,-1], [0,1,-0]])
    print("input")
    print(input)
    output = layer(input)
    print("output")
    print(output)
    print("pass-in pL/pA")
    plpa = np.random.rand(2,3)
    print(plpa)
    print("bp result")
    print(layer.backward(plpa))

def MultiLayerNetworkTest():
    input_dim = 4
    neurons = [16, 5, 3]
    activations = ["relu", "sigmoid", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)
    x = dat[:, :4]
    print("input data shape")
    print(x.shape)
    print("output data shape")
    print(net(x).shape)

def PreprocesserTest():
    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)
    print("preprocessor mean and std are: ")
    print(prep_input.min)
    print(prep_input.max)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)
    print("After preprocessing, current train data max and min: (should be 0 and 1)")
    print(x_train_pre.max(axis = 0))
    print(x_train_pre.min(axis = 0))
    print("After preprocessing, current val data max and min: (similar to 0 and 1)")
    print(x_val_pre.max(axis = 0))
    print(x_val_pre.min(axis = 0))
    
    print("Trign to revert data..")
    x_train_revert = prep_input.revert(x_train_pre)
    print("after apply")
    print(x_train_pre[0])
    print("real x_train")
    print(x_train[0])
    print("reverted x_train")
    print(x_train_revert[0])


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=100,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    # example_main()
    # LinearLayerTest()
    # ReLULayerTest()
    # SigmoidLayerTest()
    # MultiLayerNetworkTest()
    PreprocesserTest()
    # python D:\Imperial" "College" "London" "2021-2022\课程\Itro" "to" "Machine" "Learning\Courswork\intro2ml_cw2\part1_nn_lib.py