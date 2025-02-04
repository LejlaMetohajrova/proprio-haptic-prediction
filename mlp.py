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
        
        # Pick random index from the training data for plotting targets and outputs
        example = np.random.randint(X.shape[0])

        # Init weights with mean 0
        np.random.seed(1)
        self.weights0 = 2*np.random.random((X.shape[1] + 1, hidden)) -1
        self.weights1 = 2*np.random.random((hidden,Y.shape[1])) -1
        
        # Add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
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
                for j in range(14):
                    plot_tar_out(self.expected_output[example][20*j:20*(j+1)], self.output_layer[example][20*j:20*(j+1)], str(i) + '_' + str(j), ids=(20*j, 20*(j+1)))
                plot_tar_out(self.expected_output[example][20*14:], self.output_layer[example][20*14:], str(i) + '_Tact', ids=(20*14, 322))
                print(error[i])
                
            # Validation error
            self.predict(self.valid_X)
            valid_error.append(self.mse(self.valid_Y, self.output_layer))
        
        # Plot mean squared error
        f = plt.figure('Training')
        plt.plot(error, c='b', label='Training error')
        plt.plot(valid_error, c='r', label='Testing error')
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

def plot_tar_out(tar, out, name, range=None, ids=None):
    """
    Helper function to plot evaluated output compared to desired target.
    """
    axes = plt.gca()
    axes.set_ylim([-0.1,1.1])
    plt.ylabel('Activation')
    
    if range is None:
        if ids is None:
            range = np.arange(0.5, len(tar)+0.5, 1)
        else:
            range = np.arange(0.5 + ids[0], ids[1]+0.5, 1)
        plt.xlabel('Neuron ID')
    else:
        plt.xlabel('Estimated angle in degrees')

    plt.scatter(range, tar, c='r', label='Targets', s=40)
    plt.scatter(range, out, c='b', label='Output', s=40)
    plt.legend(loc=0, ncol=2, bbox_to_anchor=(0.5,-0.1))
    if ids is None:
        axes.set_xticks(np.arange(0, len(tar) + 1, 1), minor=True)
    else:
        axes.set_xticks(np.arange(ids[0], ids[1] + 1, 1), minor=True)
    plt.grid(which='minor')
    plt.savefig('pic\Figure_' + str(name), bbox_inches='tight')
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
    for j in range(14):
        plot_tar_out(tar[20*j:20*(j+1)],out[:, 20*j:20*(j+1)], 'test1_' + str(j), ids=(20*j, 20*(j+1)))
    plot_tar_out(tar[20*14:],out[:, 20*14:], 'test1_' + 'Tact', ids=(20*14, 322))
    print()
    
    print("TEST2 - on the whole dataset")
    out = p.predict(p.X)
    print("Mean squared error: ", p.mse(out, p.Y))
    tactile_error = np.array([p.mse(p.Y[i,-42:], out[i,-42:]) for i in range(len(out))])
    print()
    
    # TEST 3 - Plot mean squared error of tactile stimuli in time
    f = plt.figure('Tactile_error')
    plt.xlabel('Time')
    plt.ylabel('Mean activation')
    plt.plot(np.arange(2000,2500),np.mean(p.Y[2000:2500, -42:], axis=1), c='r', label='Desired')
    plt.plot(np.arange(2000,2500),np.mean(out[2000:2500, -42:], axis=1), c='g', label='Predicted')
    plt.legend(loc=0, ncol=3)
    f.savefig('pic\Tactile_error')
    plt.close()
    
    # TEST 4 - Plot mean squared error of tactile stimuli per neuron
    f = plt.figure('Mean activation of tactile neurons')
    plt.xlabel('Neuron ID')
    plt.ylabel('Mean activation')
    axes = plt.gca()
    axes.set_xlim(0,43)
    axes.set_ylim(-0.01,0.15)
    
    plt.scatter(np.arange(1,43, 1), np.mean(out[:,-42:], axis=0), c='g', label='Predicted')
    plt.scatter(np.arange(1,43, 1), np.mean(p.Y[:,-42:], axis=0), c='r', label='Desired')
    plt.plot(np.arange(1,43, 1), np.mean(out[:,-42:], axis=0), c='g')
    plt.plot(np.arange(1,43, 1), np.mean(p.Y[:,-42:], axis=0), c='r')
    
    axes.set_xticks(np.arange(0.5, 43.5, 1), minor=True)
    plt.grid(which='minor')
    plt.legend(loc=0, ncol=2)
    f.savefig('pic\Tactile_activation')
    plt.close()
    
    print("TEST5 - expected touch, activation=0")
    tact = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    left = np.array([-55.31235, 25.725389, 78.320732, 70.790825, -27.407681, -33.02016, -0.223165])
    right = np.array([-65.470417, 20.09986, 52.075555, 51.450444, -71.702232, -47.872209, 9.486011])
    action = np.array([0, 0, 0, 0, 0, 0, 0])
    
    gl = p.encoder.encode_gaussian(left)
    gr = p.encoder.encode_gaussian(right)
    ga = p.encoder.encode_gaussian(action, False)
    x = np.append(np.append(gl, gr), np.append(ga, ga))
    
    sl = p.encoder.encode_sigmoid(left)
    sr = p.encoder.encode_sigmoid(right)
    tar = np.append(np.append(sl, sr), tact)
    
    out = p.predict(x)
    print("Mean squared error: ", p.mse(tar, out))
    ev = p.encoder.decode_sigmoid(out[:,:-42])
    exp = p.encoder.decode_sigmoid(tar[:-42])
    print("Expected proprio:\n", exp)
    print("Evaluated proprio:\n", ev)
    print("Expected haptic:\n", tar[-42:])
    print("Evaluated haptic:\n", out[:,-42:])
    plot_tar_out(tar[20*14:], out[:, 20*14:], 'Test5', ids=(20*14, 322))
    