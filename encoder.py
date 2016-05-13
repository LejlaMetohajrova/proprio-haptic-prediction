from __future__ import print_function
import numpy as np
    
class PopulationEncoder:
    
    def __init__(self, angle_gaussians=20, action_gaussians=15, sigmoids=20):
        self.ranges = np.array([[-95, 10], [0, 160.8], [-37, 80], [15.5, 106], [-90, 90], [-90, 0], [-20, 40]])
        self.angle_gaussians = angle_gaussians
        self.action_gaussians = action_gaussians
        self.sigmoids=sigmoids
    
    def encode_gaussian(self, data, use_range=True):
        """
        Encode each value from data into a population of neurons.
        
        Input:
            data - (2D list of floats) of shape (3, number of records).
        
        Output:
            Encoded data (2D list of floats) of shape (number of records, angle_gaussians*3).
        """
        encoded = []
        
        # iterate through DoF
        for i in range(0, len(data)):
            
            # set parameters of gaussians
            if use_range:
                r = self.ranges[i%len(self.ranges)][1] - self.ranges[i%len(self.ranges)][0]
                sigma = (r / (self.angle_gaussians - 1))/2
                ni = np.array([self.ranges[i%len(self.ranges)][0] + s*sigma*2 for s in range(0, self.angle_gaussians)])
                
                for j in range(0, self.angle_gaussians):
                    encoded.append(self.gauss(data[i], ni[j], sigma))
                
            # encoding action in range(-25,25)
            else:
                sigma = (50 / (self.action_gaussians - 1))/2
                ni = np.array([-25 + s*sigma*2 for s in range(0, self.action_gaussians)])  
            
                for j in range(0, self.action_gaussians):
                    encoded.append(self.gauss(data[i], ni[j], sigma))
                
        return np.array(encoded)
    
    def gauss(self, fi, ni, sigma):
        return np.exp(- np.power((fi - ni), 2) / (2 * np.power(sigma,2)))
    
    def encode_sigmoid(self, data):
        encoded = []
        
        # iterate through DoF
        for i in range(0, len(data)):
            
            # set parameters of sigmoids
            r = self.ranges[i%len(self.ranges)][1] - self.ranges[i%len(self.ranges)][0]
            sigma = (r / (self.sigmoids - 1))/2
            ni = np.array([self.ranges[i%len(self.ranges)][0] + s*sigma*2 for s in range(0, self.sigmoids)])
            
            for j in range(0, self.sigmoids):
                encoded.append(1 / (1 + np.exp(-data[i] + ni[j])))
            
        return np.array(encoded)

    def decode_sigmoid(self, data):
        """
        Finds the neuron with activation closest to 0.5. Decodes the angle on the selected sigmoid.
        
        Input:
            data - (2D list of floats) list of shape (records, 3(DoF)*sigmoids).
            
        Output:
            Decoded data - one angle per each DoF, multiple records.
        """
        decoded = []
        
        if len(data.shape) == 1:
            data = data.reshape(1, data.shape[0])
        
        for y in data:        
            y = y.reshape(2*len(self.ranges), self.sigmoids)
            d = []
            for i in range(0, len(y)):
                min_e = 1
                index = 0
                
                for j in range(0, len(y[i])):
                    e = y[i][j] - 0.5
                    if e >= 0:
                        if e <= min_e:
                            min_e = e
                            index = j
                    else:
                        if abs(e) < min_e:
                            min_e = abs(e)
                            index = j
                
                r = self.ranges[i%len(self.ranges)][1] - self.ranges[i%len(self.ranges)][0]
                sigma = (r / (self.sigmoids - 1))/2
                ni = self.ranges[i%len(self.ranges)][0] + index*sigma*2
                
                d.append(np.log(y[i][index] / (1-y[i][index])) + ni)
            decoded.append(d)
                
        return np.array(decoded)
