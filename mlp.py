from __future__ import print_function
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from encoder import PopulationEncoder
    
class MultiLayerPerceptron:

    def __init__(self):
        self.encoder = PopulationEncoder()
        self.read_data()
        
    def read_data(self):
        """
        Initializes data from dataset and encodes it using population coding.
        """
        self.data = np.loadtxt(open("d.txt","rb"),delimiter=" ")
        
        # Divide data to haptic stimuli, proprioception of left and right hand
        haptic = np.array([x[:42] for x in self.data[:-1]])
        
        left_proprio = np.array([x[42:49] for x in self.data])
        right_proprio = np.array([x[49:] for x in self.data])
        
        # Compute action on the left and right hand (given by the difference of proprio)
        left_action = np.array([left_proprio[i+1] - left_proprio[i] for i in range(0, left_proprio.shape[0]-1)])
        right_action = np.array([right_proprio[i+1] - right_proprio[i] for i in range(0, right_proprio.shape[0]-1)])
        
        # Encode input data using gaussian population coding
        x1_left = self.encoder.encode_gaussian(left_proprio[:-1].T).T
        x2_left = self.encoder.encode_gaussian(left_action.T, False).T
        
        x1_right = self.encoder.encode_gaussian(right_proprio[:-1].T).T
        x2_right = self.encoder.encode_gaussian(right_action.T, False).T
        
        # Encode output data using sigmoid population coding
        y_left = self.encoder.encode_sigmoid(left_proprio[1:].T).T
        y_right = self.encoder.encode_sigmoid(right_proprio[1:].T).T
        
        # Prepare data for training
        self.X = np.append(np.append(x1_left, x1_right, axis=1), np.append(x2_left, x2_right, axis=1), axis=1)
        self.Y = np.append(np.append(y_left, y_right, axis=1), haptic, axis=1)

        # Randomly select and separate validation data
        self.train_X = self.X[:15000]
        self.train_Y = self.Y[:15000]
        
        self.valid_X = self.X[15000:]
        self.valid_Y = self.Y[15000:]

    def act(self, x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))
        
    def deriv(self, y):
        """
        Derivation of sigmoid activation function.
        """
        return y * (1 - y)
    
    def mse(self, ts, ys):
        """
        Mean squared error of output ys compared to target ts.
        """
        err = sum((y - t) ** 2 for (y, t) in zip(ys, ts)) / len(ys)
        if type(err) is np.ndarray and len(err) > 1:
            err = sum(err) / len(err)
        return err
        
    def eval_hidden_layer(self):
        """
        Evaluate values on the hidden layer.
        """
        self.hidden_layer = self.act(np.dot(self.input_layer, self.weights0))
        
    def eval_output_layer(self):
        """
        Evaluate values on the output layer.
        """
        self.output_layer = self.act(np.dot(self.hidden_layer, self.weights1))
            
    def train(self, X, Y, alpha=0.001, number_of_epochs=301, hidden=50):
        """
        Training.
        """
        
        #Init variables to store statistics
        error = []
        valid_error = []

        # Init weights with mean 0
        np.random.seed(1)
        self.weights0 = 2*np.random.random((X.shape[1] + 1, hidden)) -1
        self.weights1 = 2*np.random.random((hidden,Y.shape[1])) -1
        
        # Add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        # Pick random index from the training data for plotting targets and outputs
        example = np.random.randint(X.shape[0])
        
        for i in range(number_of_epochs):
            self.input_layer = X
            self.expected_output = Y
            
            #Forward-propagation
            self.eval_hidden_layer()
            self.eval_output_layer()
            
            #Back-propagation error
            output_delta = (self.expected_output - self.output_layer) * self.deriv(self.output_layer)
            hidden_delta = output_delta.dot(self.weights1.T) * self.deriv(self.hidden_layer)
            
            #Update weights
            self.weights1 += alpha * self.hidden_layer.T.dot(output_delta)
            self.weights0 += alpha * self.input_layer.T.dot(hidden_delta)
            
            # Plot targets vs. outputs
            error.append(self.mse(self.expected_output, self.output_layer))
            if i%50 == 0:
                plot_tar_out(self.expected_output[example], self.output_layer[example], i)
                print(error[i])
                
            # Validation error
            self.predict(self.valid_X)
            valid_error.append(self.mse(self.valid_Y, self.output_layer))
        
        # Plot mean squared error
        f = plt.figure('Training')
        plt.plot(error, c='b', label='Training error')
        plt.plot(valid_error, c='r', label='Validation error')
        plt.xlabel('Epochs')
        plt.ylabel('Mean squared error')
        plt.legend(loc=0, ncol=2)
        f.savefig('pic\Training')   
        plt.close()   
            
    def predict(self, in_data):
        """
        Predict output based on the given data.
        """
        if len(in_data.shape) < 2:
            in_data = in_data.reshape((1, in_data.shape[0]))
            
        self.input_layer = in_data
        
        # Add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(self.input_layer.shape[0]))
        self.input_layer = np.concatenate((ones.T, self.input_layer), axis=1)
        
        self.eval_hidden_layer()
        self.eval_output_layer()

        return self.output_layer

def plot_tar_out(tar, out, name):
    """
    Helper function to plot evaluated output compared to desired target.
    """

    t = plt.scatter(np.arange(len(tar)), tar, c='r', label='Targets')
    o = plt.scatter(np.arange(len(tar)), out, c='b', label='Output')
    
    plt.xlabel('Neuron ID')
    plt.ylabel('Activation')
    plt.legend(loc=0, ncol=2)
    plt.savefig('pic\Figure_' + str(name))
    plt.close()

if __name__ == '__main__':
    np.set_printoptions(precision=2)
    
    p = MultiLayerPerceptron()    
    p.train(p.train_X, p.train_Y)
    print()
    
    rand_index = np.random.randint(p.valid_X.shape[0])
    print("TEST1 - random sample from validation dataset with index: ", rand_index)
    out = p.predict(p.valid_X[rand_index])
    tar = p.valid_Y[rand_index]
    print("Mean squared error: ", p.mse(tar, out))
    ev = p.encoder.decode_sigmoid(out[:,:-42])
    exp = p.encoder.decode_sigmoid(tar[:-42])
    print("Expected proprio:\n", exp)
    print("Evaluated proprio:\n", ev)
    print("Expected haptic:\n", tar[-42:])
    print("Evaluated haptic:\n", out[:,-42:])
    plot_tar_out(tar, out, 'test1')
    print()
    
    print("TEST2 - on the whole dataset")
    out = p.predict(p.X)
    print("Mean squared error: ", p.mse(out, p.Y))
    tactile_error = np.array([p.mse(p.Y[i,-42:], out[i,-42:]) for i in range(len(out))])
    
    # Plot mean squared error of tactile stimuli
    f = plt.figure('Tactile_error')
    plt.plot(tactile_error, c='g')
    plt.close()
    
    f = plt.figure('Mean activation of tactile neurons')
    plt.xlabel('Neuron ID')
    plt.ylabel('Mean activation')
    plt.plot(np.mean(out[:,-42:], axis=0), c='g', label='Predicted')
    plt.plot(np.mean(p.Y[:,-42:], axis=0), c='r', label='Desired')
    plt.legend(loc=0, ncol=2)
    f.savefig('pic\Tactile_activation')
    plt.close()