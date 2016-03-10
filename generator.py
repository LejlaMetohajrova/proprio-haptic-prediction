from __future__ import print_function
import numpy as np
import pickle

class Generator:
    
    def __init__(self):
        self.read_data()
        
    def read_data(self):
        self.range = np.array([[-25, 10], [-5, 50], [20, 170]])
    
    def line(self, x, a, b):
        return a*x + b
    
    def sinus(self, x, a, ni, rg):
        return rg*np.sin(x*a) + ni

    """
    Generates random trajectory (either a line or sinus) for each degree of freedom and returns samples.
    """
    def sample(self, count):
        data = [] 
        for r in self.range:
            use_line = np.random.randint(2)
            
            rg = r[1] - r[0]
            b = r[0] + rg/2
            a = 2*np.random.random_sample() -1 #range (-1, 1)
            
            x1 = (r[0] - b)/a
            x2 = (r[1] - b)/a
            x = np.random.uniform(x1, x2, count)
            x.sort()
            
            if use_line:
                data.append(self.line(x, a, b))
            else:
                data.append(self.sinus(x, a, b, rg/2))
                
        return np.array(data)
        
    def generate(self, count, train=True):
        samples = self.sample(count+1).T
        """
        X1, X2 - (2D list of floats) features X[i][j]
            where i = count
            and j = 3 DoF (X1 - degree, X2 - action)
        Y - (2D list of floats) labels Y[i][j]
            where i = count
            and j = 3 (DoF)
        """
        X1 = []
        X2 = []
        Y = []
        #iterate through records
        for i in range(0, len(samples) -1):
            recordx1 = []
            recordx2 = []
            recordy = []
            #iterate through DoF
            for j in range(0, len(samples[i])):
                recordx1.append(samples[i][j])
                recordx2.append(samples[i+1][j] - samples[i][j])
                recordy.append(samples[i+1][j])
            X1.append(recordx1)
            X2.append(recordx2)
            Y.append(recordy)
        
        X1 = np.array(X1)
        X2 = np.array(X2)
        Y = np.array(Y)
            
        if(not train):
            pickle.dump(Y, open('exp.p', 'wb'))
            return [X1, X2]
        
        return [X1, X2, Y]
            
if __name__ == '__main__':
    g = Generator()
    
    np.set_printoptions(precision=2)
    
    pickle.dump(g.generate(300), open('data.p', 'wb'))
    pickle.dump(g.generate(10, False), open('test.p', 'wb'))