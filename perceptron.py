from __future__ import print_function
import numpy as np
import pickle
import matplotlib.pyplot as plt
from encoder import GaussianEncoder
    
class Perceptron:

    def __init__(self):
        self.encoder = GaussianEncoder()
        self.read_data()
        
    def read_data(self):
        """
        Initializes data from dataset and encodes them,
        
        X (2D array of floats) - encoded input,        
        where X[i][j]:
        
        i: number of records
        j: encoded data (for each degree of freedom and each action, there are several gaussians)
        
        Y (2D array of floats) - encoded desired output,
        where Y[i][j]:
        
        i: number of records
        j: encoded data (for each degree of freedom, there are several gaussians)
        """
        X1, X2, Y = pickle.load(open('data.p', 'rb'))
        
        x1 = self.encoder.encode_data(X1.T)
        x2 = self.encoder.encode_data(X2.T, False)
        
        self.X = np.append(x1, x2, axis=0).T
        #self.Y = self.encoder.encode_data(Y.T).T
        self.Y = Y
        
    def act(self, x):
        #return np.tanh(x)
        return 1 / (1 + np.exp(-x))
        
    def deriv(self, y):
        #return 1 - y*y
        return y * (1 - y)
    
    def mean_squared_error(self, ts, ys):
        err = sum((y - t) ** 2 for (y, t) in zip(ys, ts)) / len(ys)
        if len(err) > 1:
            err = sum(err) / len(err)
        return err
        
    def eval_hidden_layer(self):
        self.hidden_layer = self.act(np.dot(self.input_layer, self.weights0))
        
    def eval_output_layer(self):        
        self.output_layer = self.act(np.dot(self.hidden_layer, self.weights1))
            
    def train(self, X, Y, alpha=0.001, number_of_epochs=100):      
        
        error = np.array([])
    
        # Initialize weights
        np.random.seed(1)
        self.weights0 = 2*np.random.random((3*self.encoder.number_of_gauss + 3*self.encoder.action_number_of_gauss + 1, 5)) - 1
        #self.weights1 = 2*np.random.random((100, 3*self.encoder.number_of_gauss)) - 1
        self.weights1 = 2*np.random.random((5, 3)) - 1
        
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        for i in range(number_of_epochs):
            
            perm = np.random.permutation(len(X))     
            self.input_layer = X[perm]
            self.expected_output = Y[perm]
            
            self.eval_hidden_layer()
            self.eval_output_layer()
            
            output_delta = (self.expected_output - self.output_layer) * self.deriv(self.output_layer)
            hidden_delta = output_delta.dot(self.weights1.T) * self.deriv(self.hidden_layer)
            
            self.weights1 += alpha * self.hidden_layer.T.dot(output_delta)
            self.weights0 += alpha * self.input_layer.T.dot(hidden_delta)
            
            # Weight decay
            self.weights0 *= 0.999999
            self.weights1 *= 0.999999
            
            # Plot mean squared error
            error = np.append(error, self.mean_squared_error(self.expected_output, self.output_layer))
            plt.plot(error)
            plt.draw()
            
            if i%100 == 0:
                print(error[i])
        
        plt.show()
            
    def predict(self, in_data, encode=True):
        """
        Predicts output based on the given data.
        Can be used in either encoded or unencoded mode.
        
        Input:
            if encode:
                in_data - 3D list of floats of shape (2 (degree, action),records, 3 (DoF))
            else:
                in_data - 2D list of floats of shape (records, 2*3*number_of_gauss)
                
        Output:
            if encode:
                decoded output
            else:
                2D list of floats of shape (records, 3(DoF)*number_of_gauss)
        """
        if(encode):
            x1 = self.encoder.encode_data(in_data[0].T)
            x2 = self.encoder.encode_data(in_data[1].T)
            self.input_layer = np.append(x1, x2, axis = 0).T
        else:
            self.input_layer = in_data
        
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(self.input_layer.shape[0]))
        self.input_layer = np.concatenate((ones.T, self.input_layer), axis=1)
        
        self.eval_hidden_layer()
        self.eval_output_layer()
        
        if(encode):
            return self.encoder.decode_data(self.output_layer.T)
        else:
            return self.output_layer

if __name__ == '__main__':
    np.set_printoptions(precision=2)
    
    p = Perceptron()
    
    p.train(p.X, p.Y/100.0)
    #pickle.dump([p.weights0, p.weights1], open('weights.p', 'wb'))
    #p.weights0, p.weights1 = pickle.load(open('weights.p', 'rb'))
    
    print("TEST1")
    deg = p.encoder.encode_data(np.array([[17, 39, 50]]).T)
    act = p.encoder.encode_data(np.array([[0.5, -0.8, -0.2]]).T, False)
    out = p.predict(np.append(deg, act, axis=0).T, False)
    tar =  np.array([[17, 39, 50]])/ 100.0
    print(p.mean_squared_error(tar, out))
    
    print("Expected: ", np.array([[17, 39, 50]]))
    print("Evaluated: ", out*100)
