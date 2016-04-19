from __future__ import print_function
import numpy as np
    
class GaussianEncoder:
    
    def __init__(self, number_of_gauss=22, action_number_of_gauss=6):
        """
        Input:
            number_of_gauss - number of gaussians to be used for encoding a value.
        """
        self.ranges = np.array([[-95, 10], [0, 160.8], [-37, 80], [15.5, 106], [-90, 90], [-90, 0], [-20, 40]])
        self.number_of_gauss = number_of_gauss
        self.action_number_of_gauss = action_number_of_gauss
        self.sigmoids=20
    
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
                
                for j in range(0, self.number_of_gauss):
                    encoded.append(self.gauss(data[i], ni[j], sigma))
                
            # encoding action in range(-10,10)
            else:
                sigma = (20 / (self.number_of_gauss - 1))/2
                ni = np.array([-10 + s*sigma*2 for s in range(0, self.action_number_of_gauss)])  
            
                for j in range(0, self.action_number_of_gauss):
                    encoded.append(self.gauss(data[i], ni[j], sigma))
                
        return np.array(encoded)
        
    def encode_targets(self, data):
        encoded = []
        
        # iterate through DoF
        for i in range(0, len(data)):
            
            # set parameters of sigmoids
            r = self.ranges[i][1] - self.ranges[i][0]
            sigma = (r / (self.sigmoids - 1))/2
            ni = np.array([self.ranges[i][0] + s*sigma*2 for s in range(0, self.sigmoids)])
            
            for j in range(0, self.sigmoids):
                encoded.append(1 / (1 + np.exp(-data[i] + ni[j])))
            
        return np.array(encoded)
                
    def gauss(self, fi, ni, sigma):
        return np.exp(- np.power((fi - ni), 2) / (2 * np.power(sigma,2)))
                
    def decode_data(self, data):
        """
        Decode angles from population of neurons. For each activation of gaussian it returns 2 angles.
        
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
        
    def maxSumIndexes(self, population):
        """
        Input:
            population - array of activations of a population of gaussians.
            
        Output:
            Returns two adjacent indexes, which sum of values is maximum.
        """
        
        maxIndex = np.zeros(population.shape[1])
        maxValue = np.zeros(population.shape[1])
        
        for i in range(0, len(population) - 1):
            sum = population[i] + population[i+1]
            
            greater = sum > maxValue
            for j in range(0, len(greater)):
                if greater[j]:
                    maxValue[j] = sum[j]
                    maxIndex[j] = i
        
        return np.array([maxIndex, maxIndex + 1]).astype(int)
        
    def decode_max_sum_data(self, data):
        """
        Decode angles from population of neurons, which fire the most. Returns one angle per population.
        
        Input:
            data - (2D list of floats) list of shape (records, 3(DoF)*number_of_gauss).
            
        Output:
            Decoded data - one angle per each DoF, multiple records.
        """
        decoded = []
        
        for y in data:
            # 3 DoF x number_of_gauss            
            y = y.reshape(self.number_of_gauss, 3)
            indexes = self.maxSumIndexes(y).T
            d = []
            for i in range(0, len(indexes)):
                # set parameters of gaussians
                r = self.ranges[i][1] - self.ranges[i][0]
                sigma = (r / (self.number_of_gauss - 1))/2
                ni = np.array([self.ranges[i][0] + s*sigma*2 for s in indexes[i]])     
                
                d.append(np.median(self.inverse_gauss(y.T[i][indexes[i]], ni, sigma)))
            decoded.append(d)
                
        return np.array(decoded)
        
    def decode_eval_data(self, data):
        """
        Decode angles from neurons in populations, which fire the most. Returns one angle per population.
        
        Input:
            data - (2D list of floats) list of shape (records, 3(DoF)*number_of_gauss).
            
        Output:
            Decoded data - one angle per each DoF, multiple records.
        """
        decoded = []
        
        for y in data:
            # 3 DoF x number_of_gauss            
            y = y.reshape(len(self.ranges), self.number_of_gauss)
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
        
    def decode_sigmoids(self, data):
        """
        Finds the neuron with activation closest to 0.5. Decodes the angle on the selected sigmoid.
        
        Input:
            data - (2D list of floats) list of shape (records, 3(DoF)*sigmoids).
            
        Output:
            Decoded data - one angle per each DoF, multiple records.
        """
        decoded = []
        
        for y in data:        
            y = y.reshape(len(self.ranges), self.sigmoids)
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
                
                r = self.ranges[i][1] - self.ranges[i][0]
                sigma = (r / (self.sigmoids - 1))/2
                ni = self.ranges[i][0] + index*sigma*2
                
                d.append(np.log(y[i][index] / (1-y[i][index])) + ni)
            decoded.append(d)
                
        return np.array(decoded)
