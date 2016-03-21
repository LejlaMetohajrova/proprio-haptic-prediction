from __future__ import print_function
import numpy as np
import pickle
import matplotlib.pyplot as plt
    
class Perceptron:

    def __init__(self):
        self.read_data()
        
    def read_data(self):        
        self.X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
        self.Y = np.array([[0,1,1,0]]).T
        
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
            
    def train(self, X, Y, alpha=0.3, number_of_epochs=10001):      
        
        error = np.array([])
    
        # Initialize weights
        np.random.seed(1)
        self.weights0 = 2*np.random.random((3, 5)) - 1
        self.weights1 = 2*np.random.random((5, 1)) - 1
        
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        #ones = np.atleast_2d(np.ones(X.shape[0]))
        #X = np.concatenate((ones.T, X), axis=1)
        
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
            
            if i%1000 == 0:
                print(error[i])
                
        plt.plot(error)
        plt.show()
            
    def predict(self, in_data):
        
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
    # 
    # print("TEST1")
    # deg = p.encoder.encode_data(np.array([[17, 39, 50]]).T)
    # act = p.encoder.encode_data(np.array([[0.5, -0.8, -0.2]]).T, False)
    # out = p.predict(np.append(deg, act, axis=0).T, False)
    # tar =  np.array([[17.5, 38.2, 49.8]])/ 100.0
    # print(p.mean_squared_error(tar, out))
    # 
    # print("Expected: ", np.array([[17.5, 38.2, 49.8]]))
    # print("Evaluated: ", out*100)
    # 
    # X1, X2, Y = pickle.load(open('data.p', 'rb'))    
    # x1 = p.encoder.encode_data(X1[0:1].T)
    # x2 = p.encoder.encode_data(X2[0:1].T, False)
    # out = p.predict(np.append(x1, x2, axis=0).T, False)
    # print(p.mean_squared_error(Y[0:1]/100.0, out))
    # print("Expected: ", Y[0:1])
    # print("Evaluated: ", out*100)
