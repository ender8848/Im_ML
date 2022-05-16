import torch
import pickle
import numpy as np
from numpy.random import default_rng
import pandas as pd
from sklearn import preprocessing
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import sklearn.metrics as sm
import warnings
import random
warnings.filterwarnings("ignore")

class Regressor(nn.Module):

    def __init__(self,
                 x,
                 nb_epoch=1000,
                 batch_size=100,
                 lr=0.001,
                 textual_value='ocean_proximity',
                 neurons=[8,8,8],
                 activations=["relu", "relu","relu"],
                 shuffle = True
                 ):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Initialize member variables according to parameters
        super(Regressor, self).__init__()
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.textual_value = textual_value
        self.shuffle = shuffle

        if neurons is not None:
            self.neurons = neurons
        if activations is not None:
            self.activations = activations

        # Preprocess data and calculate input and output size
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.normalizer = None

        # Build a neural network based on default parameters or specified parameters
        assert (len(neurons) == len(activations))

        idx = 0
        for i in range(len(neurons)):
            if i == 0:
                self._modules[str(idx)] = nn.Linear(self.input_size, neurons[i])
                idx += 1
            else:
                self._modules[str(idx)] = nn.Linear(neurons[i - 1], neurons[i])
                idx += 1
            if activations[i] == "sigmoid":
                self._modules[str(idx)] = nn.Sigmoid()
                idx += 1
            if activations[i] == "relu":
                self._modules[str(idx)] = nn.ReLU()
                idx += 1
            if activations[i] == "tanh":
                self._modules[str(idx)] = nn.Tanh()
                idx += 1

        self._modules[str(idx)] = nn.Linear(neurons[-1], self.output_size)

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

        # Handle the missing values in the data, setting them to mean values
        x = x.fillna(x.mean())
        if y is not None:
            y = y.fillna(y.mean())

        # Handle the textual values in the data, encoding them using one-hot encoding
        onehot = pd.get_dummies(x[self.textual_value])
        onehot = onehot.values
        x.drop(self.textual_value, axis=1, inplace=True)
        x = x.values

        # Apply max-min normalization
        if training:
            normalizer = preprocessing.MinMaxScaler()
            x = normalizer.fit_transform(x)
            self.normalizer = normalizer
        else:
            x = self.normalizer.transform(x)

        # Add ont-hot data to the dataset
        x = np.concatenate((x, onehot), axis=1)

        return x.astype(float), (y.values.astype(float) if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget

        training_data = TensorDataset(torch.Tensor(X), torch.Tensor(Y))
        training_dataloader = DataLoader(
            training_data, batch_size=self.batch_size, shuffle=self.shuffle
        )

        #
        loss_curve_final = []
        for epoch in range(self.nb_epoch):
            loss_curve = []
            for inputs, labels in training_dataloader:
                #
                preds = self(inputs)
                loss= nn.MSELoss()
                lossResult = loss(preds, labels)
                self.optimizer.zero_grad()
                lossResult.backward()
                self.optimizer.step()

                #
                loss_curve += [lossResult.item()]
                loss_curve_final += [lossResult.item()]
            #
            # print('--- Iteration {0}: training loss = {1:.4f} ---'.format(epoch + 1, np.array(loss_curve_final).mean()))
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=False)  # Do not forget
        with torch.no_grad():
            preds = self(torch.Tensor(X)).detach().numpy()

        return preds
        
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y=y, training=False)  # Do not forget
        preds = self.predict(x)
        mse = sm.mean_squared_error(Y, preds)
        mape = sm.mean_absolute_percentage_error(Y, preds)
        mae = sm.mean_absolute_error(Y, preds)
        # print("mean absolute error is :")
        # print(mae)
        # print("absolute mean percentage error is :")
        # print(mape)

        return mse

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

class Train_val_test:
    
    def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
        """ Split n_instances into n mutually exclusive splits at random.
        
        Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator
        
        Returns:
        list: a list (length n_splits). Each element in the list should contain a 
        numpy array giving the indices of the instances in that split.
        """
        # generate a random permutation of indices from 0 to n_instances
        shuffled_indices = random_generator.permutation(n_instances)
        # split shuffled indices into almost equal sized splits
        split_indices = np.array_split(shuffled_indices, n_splits)
        return split_indices
    
    def train_test_val_idx(n_folds, n_instances, random_generator=default_rng()):
        """ Generate train and test indices at each fold.
        
        Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

        Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
        with two elements: a numpy array containing the train indices, and another 
        numpy array containing the test indices.
        """

        # split the dataset into train val and test splits
        split_indices = Train_val_test.k_fold_split(n_folds, n_instances, random_generator)
        test_idx = split_indices[0]
        val_idx = split_indices[1]
        train_idx = []
        for i in range(n_folds - 2):
            train_idx = train_idx + split_indices[i].tolist()
        return train_idx, val_idx, test_idx


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def generate_neurons_list(layers_lb = 1, layers_ub = 5, size_lb = 5, size_ub = 20, total_count = 20):
    result = []
    for i in range(total_count):
        layers = random.randint(layers_lb, layers_ub)
        temp = []
        for j in range(layers):
            r = random.randint(size_lb, size_ub)
            temp.append(r)
        result.append(temp)
    return result

def RegressorHyperParameterSearch():
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    
    data = pd.read_csv("housing.csv")
    output_label = "median_house_value"

    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    train_idx, val_idx, test_idx = Train_val_test.train_test_val_idx(10, len(y.index))
    x_train = pd.DataFrame(x, index = train_idx)
    x_val   = pd.DataFrame(x, index = val_idx)
    x_test  = pd.DataFrame(x, index = test_idx)
    y_train = pd.DataFrame(y, index = train_idx)
    y_val   = pd.DataFrame(y, index = val_idx)
    y_test  = pd.DataFrame(y, index = test_idx)

    while (x_train.loc[:, x_train.columns == "ocean_proximity"].nunique()[0] != x_val.loc[:, x_val.columns == "ocean_proximity"].nunique()[0]) or\
    (x_train.loc[:, x_train.columns == "ocean_proximity"].nunique()[0] != x_test.loc[:, x_test.columns == "ocean_proximity"].nunique()[0]):
        train_idx, val_idx, test_idx = Train_val_test.train_test_val_idx(10, len(y.index))
        x_train = pd.DataFrame(x, index = train_idx)
        x_val   = pd.DataFrame(x, index = val_idx)
        x_test  = pd.DataFrame(x, index = test_idx)
        y_train = pd.DataFrame(y, index = train_idx)
        y_val   = pd.DataFrame(y, index = val_idx)
        y_test  = pd.DataFrame(y, index = test_idx)

    regressor = None
    # Choose the optimal activation function
    min_mse = float("inf")
    optimal_function = None
    for current_function in ["sigmoid", "relu", "tanh"]:       
        regressor = Regressor(
            x_train,
            nb_epoch=100,
            batch_size=100,
            lr=0.001,
            textual_value='ocean_proximity',
            neurons=[8, 8],
            activations=[current_function for i in range(2)],
            shuffle = True
        )
        regressor.fit(x_train, y_train)
        
        # Calculate the mse of the current generation
        current_mse = regressor.score(x_val, y_val)
        print("[Current mean squared error]:", current_mse, ",", "[activation function]:", current_function)

        if current_mse < min_mse:
            min_mse = current_mse
            optimal_function = current_function
    print("[Optimal activation function]:", optimal_function)

    # Choose the optimal number of neurons and their size
    min_mse = float("inf")
    optimal_neurons_list = None
    neurons_list = generate_neurons_list()
    for current_neurons_list in neurons_list:       
        regressor = Regressor(
            x_train,
            nb_epoch=100,
            batch_size=100,
            lr=0.001,
            textual_value='ocean_proximity',
            neurons=current_neurons_list,
            activations=[optimal_function for i in range(len(current_neurons_list))],
            shuffle = True
        )
        regressor.fit(x_train, y_train)
        
        # Calculate the mse of the current generation
        current_mse = regressor.score(x_val, y_val)
        print("[Current mean squared error]:", current_mse, ",", "[neurons]:", current_neurons_list)

        if current_mse < min_mse:
            min_mse = current_mse
            optimal_neurons_list = current_neurons_list
    print("[Optimal neurons]:", optimal_neurons_list)

    # Choose the optimal batch size
    min_mse = float("inf")
    optimal_batch_size = None
    for current_batch_size in [10, 100, 200, 400, 800, 1000]:       
        regressor = Regressor(
            x_train,
            nb_epoch=100,
            batch_size=100,
            lr=0.001,
            textual_value='ocean_proximity',
            neurons=optimal_neurons_list,
            activations=[optimal_function for i in range(len(optimal_neurons_list))],
            shuffle = True
        )
        regressor.fit(x_train, y_train)
        
        # Calculate the mse of the current generation
        current_mse = regressor.score(x_val, y_val)
        print("[Current mean squared error]:", current_mse, ",", "[batch size]:", current_batch_size)

        if current_mse < min_mse:
            min_mse = current_mse
            optimal_batch_size = current_batch_size
    print("[Optimal batch size]:", optimal_batch_size)

    # Choose the optimal learning rate
    min_mse = float("inf")
    optimal_learing_rate = None
    for current_learing_rate in [0.001, 0.01, 0.1, 1, 10]:       
        regressor = Regressor(
            x_train,
            nb_epoch=100,
            batch_size=100,
            lr=current_learing_rate,
            textual_value='ocean_proximity',
            neurons=optimal_neurons_list,
            activations=[optimal_function for i in range(len(optimal_neurons_list))],
            shuffle = True
        )
        regressor.fit(x_train, y_train)
        
        # Calculate the mse of the current generation
        current_mse = regressor.score(x_val, y_val)
        print("[Current mean squared error]:", current_mse, ",", "[learning rate]:", current_learing_rate)

        if current_mse < min_mse:
            min_mse = current_mse
            optimal_learing_rate = current_learing_rate
    print("[Optimal learning rate]:", optimal_learing_rate)

    # Choose the optimal epoch
    min_mse = float("inf")
    optimal_epoch = None
    for current_epoch in [10, 100, 200, 400, 800, 1000]:     
        regressor = Regressor(
            x_train,
            nb_epoch=current_epoch,
            batch_size=optimal_batch_size,
            lr=optimal_learing_rate,
            textual_value='ocean_proximity',
            neurons=optimal_neurons_list,
            activations=[optimal_function for i in range(len(optimal_neurons_list))],
            shuffle = True
        )
        regressor.fit(x_train, y_train)
        
        # Calculate the mse of the current generation
        current_mse = regressor.score(x_val, y_val)
        print("[Current mean squared error]:", current_mse, ",", "[epoch size]:", current_epoch)

        if current_mse < min_mse:
            min_mse = current_mse
            optimal_epoch = current_epoch
    print("[Optimal epoch]:", optimal_epoch)
    print("Test:")
    print(regressor.score(x_test, y_test))
    return optimal_function, optimal_neurons_list, optimal_batch_size , optimal_learing_rate, optimal_epoch
    # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))

def RegressorHyperParameterSearchCanRunTest():
    result = RegressorHyperParameterSearch()
    print("Result:")
    print(result)

if __name__ == "__main__":
    #example_main()
    RegressorHyperParameterSearchCanRunTest()

