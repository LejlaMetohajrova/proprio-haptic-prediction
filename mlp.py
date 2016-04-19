from __future__ import print_function
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from encoder import GaussianEncoder

def plot_tar_out(tar, out, i):
    s = tar.shape[1]
    f, axarr = plt.subplots(3, sharex=True)
    
    for j in range(0, 3):
        t = axarr[j].scatter(np.arange(s), tar[5000*j], c='r')
        o = axarr[j].scatter(np.arange(s), out[5000*j], c='b')
        axarr[j].set_title('Record no.: ' + str(5000*j))
    
    #plt.legend((t, o), ('Targets', 'Output'))
    f.savefig('pic\Figure' + str(i))
    plt.close()
    
class Perceptron:

    def __init__(self):
        self.encoder = GaussianEncoder()
        self.read_data()
        
    def read_data(self):
        """
        Initializes data from dataset and encodes them.
        """
        data = np.loadtxt(open("d.txt","rb"),delimiter=" ")
        
        haptic = np.array([x[:40] for x in data[:-1]])
        
        left_proprio = np.array([x[40:47] for x in data])
        right_proprio = np.array([x[47:] for x in data])
        
        left_action = np.array([left_proprio[i+1] - left_proprio[i] for i in range(0, left_proprio.shape[0]-1)])
        right_action = np.array([right_proprio[i+1] - right_proprio[i] for i in range(0, right_proprio.shape[0]-1)])
        
        x1_left = self.encoder.encode_data(left_proprio[:-1].T).T
        x2_left = self.encoder.encode_data(left_action.T, False).T
        
        x1_right = self.encoder.encode_data(right_proprio[:-1].T).T
        x2_right = self.encoder.encode_data(right_action.T, False).T
        
        y_left = self.encoder.encode_targets(left_proprio[1:].T).T
        y_right = self.encoder.encode_targets(right_proprio[1:].T).T
        
        X = np.append(np.append(x1_left, x1_right, axis=1), np.append(x2_left, x2_right, axis=1), axis=1)
        Y = np.append(np.append(y_left, y_right, axis=1), haptic, axis=1)

        perm = np.random.permutation(len(X))
        self.X = X[perm[:18000]]
        self.Y = Y[perm[:18000]]
        
        self.VX = X[perm[18000:]]
        self.VY = Y[perm[18000:]]


    def act(self, x, type='sigmoid'):
        if type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            return np.tanh(x)
        
    def deriv(self, y, type='sigmoid'):
        if type == 'sigmoid':
            return y * (1 - y)
        else:
            return 1 - y*y
    
    def mean_squared_error(self, ts, ys):
        err = sum((y - t) ** 2 for (y, t) in zip(ys, ts)) / len(ys)
        if len(err) > 1:
            err = sum(err) / len(err)
        return err
        
    def eval_hidden_layer(self):
        self.hidden_layer = self.act(np.dot(self.input_layer, self.weights0))
        
    def eval_output_layer(self):        
        self.output_layer = self.act(np.dot(self.hidden_layer, self.weights1))
            
    def train(self, X, Y, alpha=0.001, number_of_epochs=201, hidden=50, momentum=0):
        
        error = []
        valid_error = []
        
        self.gradient0 = 0
        self.gradient1 = 0
    
        # Initialize weights
        np.random.seed(1)
        self.weights0 = 2*np.random.random((2*(len(self.encoder.ranges)*self.encoder.number_of_gauss + len(self.encoder.ranges)*self.encoder.action_number_of_gauss) + 1, hidden)) -1
        self.weights1 = 2*np.random.random((hidden, 2*len(self.encoder.ranges)*self.encoder.sigmoids + 40)) -1
        
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        for i in range(number_of_epochs):
            
            self.input_layer = X
            self.expected_output = Y
            
            self.eval_hidden_layer()
            self.eval_output_layer()
            
            output_delta = (self.expected_output - self.output_layer) * self.deriv(self.output_layer)
            hidden_delta = output_delta.dot(self.weights1.T) * self.deriv(self.hidden_layer)
            
            # Momentum
            self.gradient1 = alpha * self.hidden_layer.T.dot(output_delta) + self.gradient1 * momentum
            self.gradient0 = alpha * self.input_layer.T.dot(hidden_delta) + self.gradient0 * momentum
            
            self.weights1 += self.gradient1
            self.weights0 += self.gradient0
            
            # Weight decay
            #self.weights0 *= 0.999999
            #self.weights1 *= 0.999999
            
            # Plot targets vs. outputs
            error.append(self.mean_squared_error(self.expected_output, self.output_layer))
            if i%10 == 0:
                plot_tar_out(self.expected_output, self.output_layer, i)
                print(error[i])
                
            # Validation
            self.predict(self.VX)
            valid_error.append(self.mean_squared_error(self.VY, self.output_layer))
        
        # Plot mean squared error
        f = plt.figure('Training')
        plt.plot(error, c='b')
        plt.plot(valid_error, c='r')
        f.savefig('pic\Training')      
            
    def predict(self, in_data):
        """
        Predicts output based on the given data.
        """
        self.input_layer = in_data
        
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(self.input_layer.shape[0]))
        self.input_layer = np.concatenate((ones.T, self.input_layer), axis=1)
        
        self.eval_hidden_layer()
        self.eval_output_layer()

        return self.output_layer

if __name__ == '__main__':
    np.set_printoptions(precision=2)
    
    p = Perceptron()
    
    p.train(p.X, p.Y)
    #pickle.dump([p.weights0, p.weights1], open('weights.p', 'wb'))
    #p.weights0, p.weights1 = pickle.load(open('weights.p', 'rb'))
    '''
    data = np.loadtxt(open("leftArm.log","rb"),delimiter=" ", usecols=range(3,10))
    X1 = data[:-1]
    X2 = np.array([data[i+1] - data[i] for i in range(0, len(data)-1)])
    Y = np.array([data[i] for i in range(1, len(data))])
    
    x1 = p.encoder.encode_data(X1.T)
    x2 = p.encoder.encode_data(X2.T, False)
    
    print("TEST1")
    x1 = p.encoder.encode_data(X1[10:11].T)
    x2 = p.encoder.encode_data(X2[10:11].T, False)
    out = p.predict(np.append(x1, x2, axis=0).T, False)
    tar = p.encoder.encode_targets(Y[10:11].T).T
    print(p.mean_squared_error(tar, out))
    o2 = p.encoder.decode_sigmoids(out)
    print("Expected:\n", Y[10:11])
    print("Evaluated:\n", o2)
    print()
 
    print("TEST2")
    x1 = p.encoder.encode_data(X1[211:212].T)
    x2 = p.encoder.encode_data(X2[211:212].T, False)
    out = p.predict(np.append(x1, x2, axis=0).T, False)
    tar = p.encoder.encode_targets(Y[211:212].T).T
    print(p.mean_squared_error(tar, out))
    o2 = p.encoder.decode_sigmoids(out)
    print("Expected:\n", Y[211:212])
    print("Evaluated:\n", o2)
    print()
    '''
    