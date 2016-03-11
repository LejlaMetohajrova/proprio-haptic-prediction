from __future__ import print_function
from random import random
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
    
class GaussianEncoder:
    def __init__(self, number_of_gauss=10):
        """
        Input:
            number_of_gauss - number of gaussians to be used for encoding a value.
        """
        self.ranges = np.array([[-25, 10], [-5, 50], [20, 170]])
        self.number_of_gauss = number_of_gauss
    
    def encode_data(self, data, use_range=True):
        """
        Encode each value from data into a population of neurons.
        
        Input:
            data - (2D list of floats) of shape (3, number of records).
        
        Output:
            Encoded data (2D list of floats) of shape (number of records, number_of_gauss*3).
        """
        encoded = []
        
        # iterate through DoF
        for i in range(0, len(data)):
            
            # set parameters of gaussians
            if use_range:
                r = self.ranges[i][1] - self.ranges[i][0]
                sigma = (r / (self.number_of_gauss - 1))/2
                ni = np.array([self.ranges[i][0] + s*sigma*2 for s in range(0, self.number_of_gauss)])
                
            # encoding action in range(-10,10)
            else:
                sigma = (20 / (self.number_of_gauss - 1))/2
                ni = np.array([-10 + s*sigma*2 for s in range(0, self.number_of_gauss)])  
            
            for j in range(0, self.number_of_gauss):
                encoded.append(self.gauss(data[i], ni[j], sigma))
                
        return np.array(encoded)
                
    def gauss(self, fi, ni, sigma):
        return np.exp(- np.power((fi - ni), 2) / (2 * np.power(sigma,2)))
                
    def decode_data(self, data):
        """
        Decode data from population of neurons.
        
        Input:
            data - (2D list of floats) list of shape (3(DoF)*number_of_gauss, records).
            
        Output:
            Decoded data (3D list of floats) of shape (data.shape[0], data.shape[1], 2).
        """
        decoded = []
        
        for i in range(0, len(data)):
            if i%self.number_of_gauss == 0:
                # set parameters of gaussians
                r = self.ranges[int(i/ self.number_of_gauss)][1] - self.ranges[int(i/ self.number_of_gauss)][0]
                sigma = (r / (self.number_of_gauss - 1))/2
                ni = np.array([self.ranges[int(i/ self.number_of_gauss)][0] + s*sigma*2 for s in range(0, self.number_of_gauss)])                
            
            decoded.append(np.array(self.inverse_gauss(data[i], ni[i%self.number_of_gauss], sigma)).T)
                
        return np.array(decoded)
        
    def inverse_gauss(self, y, ni, sigma):
        a = - np.sqrt(np.absolute(2* np.power(sigma,2) * np.log(y))) + ni
        b = np.sqrt(np.absolute(2* np.power(sigma,2) * np.log(y))) + ni
        return (a, b)
        
    def decode_eval_data(self, data):
        """
        Decode data from population of neurons. which fire enough.
        
        Input:
            data - (2D list of floats) list of shape (records, 3(DoF)*number_of_gauss).
            
        Output:
            Decoded data 
        """
        decoded = []
        
        for y in data:
            # 3 DoF x number_of_gauss            
            y = y.reshape(3, 10)
            indexes = np.argpartition(y, -2).T[-2:].T # computes indexes of 2 largest values for each DoF
            d = []
            for i in range(0, len(indexes)):
                # set parameters of gaussians
                r = self.ranges[i][1] - self.ranges[i][0]
                sigma = (r / (self.number_of_gauss - 1))/2
                ni = np.array([self.ranges[i][0] + s*sigma*2 for s in indexes[i]])     
                
                d.append(np.median(self.inverse_gauss(y[i][indexes[i]], ni, sigma)))
            decoded.append(d)
                
        return np.array(decoded)

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
        
        self.X = np.append(x1, x2).reshape(x1.shape[0] + x2.shape[0], x1.shape[1]).T
        self.Y = self.encoder.encode_data(Y.T).T
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def deriv(self, y):
        return y * (1 - y)
        
    def eval_hidden_layer(self):
        self.hidden_layer = self.sigmoid(np.dot(self.input_layer, self.weights0))
        
    def eval_output_layer(self):        
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights1))
            
    def train(self, X, Y, alpha=0.001, number_of_epochs=100000):      
    
        #initialize weights
        np.random.seed(1)
        self.weights0 = 2*np.random.random((2*3*self.encoder.number_of_gauss + 1, 100)) - 1
        self.weights1 = 2*np.random.random((100, 3*self.encoder.number_of_gauss)) - 1
        
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        for i in range(number_of_epochs):
            
            X, Y = shuffle(X, Y)            
            self.input_layer = X
            self.expected_output = Y
            
            self.eval_hidden_layer()
            self.eval_output_layer()
            
            output_delta = (self.expected_output - self.output_layer) * self.deriv(self.output_layer)
            hidden_delta = output_delta.dot(self.weights1.T) * self.deriv(self.hidden_layer)
            
            self.weights1 += alpha * self.hidden_layer.T.dot(output_delta)
            self.weights0 += alpha * self.input_layer.T.dot(hidden_delta)
            
            #print actual mean squared error    
            if i%10000 == 1:
                print(mean_squared_error(self.expected_output, self.output_layer))
            
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
            self.input_layer = np.append(x1, x2).reshape(x1.shape[0] + x2.shape[0], x1.shape[1]).T
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
    
    p.train(p.X, p.Y)
    
    print("TEST1")
    X1, X2, Y = pickle.load(open('data.p', 'rb'))
    x1 = p.encoder.encode_data(X1[:1].T)
    x2 = p.encoder.encode_data(X2[:1].T, False)
    out=p.predict(np.append(x1, x2).reshape(x1.shape[0] + x2.shape[0], x1.shape[1]).T, False)
    y = p.encoder.encode_data(Y[:1].T).T
    print(mean_squared_error(y, out)) #this is a small value
    d = p.encoder.decode_eval_data(out)
    print("Expected: ", Y[:1])
    print("Evaluated: ", d)
    print()
    
    print("TEST1.1")
    X1, X2, Y = pickle.load(open('data.p', 'rb'))
    x1 = p.encoder.encode_data(X1[:5].T)
    x2 = p.encoder.encode_data(X2[:5].T, False)
    outt=p.predict(np.append(x1, x2).reshape(x1.shape[0] + x2.shape[0], x1.shape[1]).T, False)
    yt = p.encoder.encode_data(Y[:5].T).T
    print(mean_squared_error(yt, outt))
    dt = p.encoder.decode_eval_data(outt)
    print("Expected: ", Y[:5])
    print("Evaluated: ", dt)
    print()
    
    '''
    print("TEST2")
    out1=p.predict(np.array([[[-17, 39, 150]], [[0.5, -0.8, -0.2]]]))
    out1=out1.reshape((30,2))
    print("Expected: -16.5, 38.2, 149.08\n")
    print()
    '''
    
    print("TEST3")
    deg = p.encoder.encode_data(np.array([[-17, 39, 150]]).T)
    act = p.encoder.encode_data(np.array([[0.5, -0.8, -0.2]]).T, False)
    out2=p.predict(np.append(deg, act).reshape(deg.shape[0] + act.shape[0], deg.shape[1]).T, False)
    exp2=p.encoder.encode_data(np.array([[-16.5, 38.2, 149.08]]).T).T
    print(mean_squared_error(exp2, out2))
    dec = p.encoder.decode_eval_data(out2)
    print("Expected: ", np.array([[-16.5, 38.2, 149.08]]))
    print("Evaluated: ", dec)
    print()
    '''
    print("TEST4")
    print(p.predict(np.array(pickle.load(open('test.p', 'rb')))))
    print("Expected:")
    print(np.array(pickle.load(open('exp.p', 'rb'))))
    
    print("TEST - encoding + decoding")
    a = p.encoder.encode_data(np.array([[3, 21, 165]]).T)
    out= p.encoder.decode_data(a)
    print("Expected: 3, 21, 165\n")
    '''