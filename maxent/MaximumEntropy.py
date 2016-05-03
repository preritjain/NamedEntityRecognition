'''
Created on 03-May-2016

@author: prerit
'''
import csv
import numpy as np
from maxent.Datum import Datum
class MaximumEntropy:
    '''
    classdocs
    '''


    def __init__(self):
        self.labels = []
        self.labelIndexes = {}
        self.features = []
        self.featureIndexes = {}
        self.lambdas = []
        '''
        Constructor
        '''
    def readTrainFile(self,filename):
        csv_reader = csv.DictReader(open(filename,'r'))
        data = []
        prevLabel = 'O'
        for row in csv_reader:
            
            #print("row len ",len(row))
            datum = Datum(row['word'],row['label'])
            for i in range(1,len(row)-1):
                k = 'g'+str(i)
                datum.features[k]=row[k]
            
            datum.previousLabel = prevLabel
            prevLabel = datum.label
            data.append(datum)
            #print(self.labelIndexes)
            if self.labelIndexes.get(datum.label) == None:
                lIndex = len(self.labels)
                self.labels.append(datum.label)
                self.labelIndexes[datum.label] = lIndex
        
            
        feat = list(data[0].features.keys())
        self.features = feat
        print("feat", self.featureIndexes)
        for i in range(0,len(feat)):
            self.featureIndexes[feat[i]] = i  
        
        self.lambdas = np.zeros(len(self.features))
        return data  
            
            
    def empericalCount(self,trainFeatures):
        
        weights = np.zeros(shape=(len(self.labels),len(trainFeatures[0].features)))
        for datum in trainFeatures:
            for label in self.labels:
                for feature in self.features:
                    if (label.strip() != 'O') and (datum.label == label) and (datum.features[feature]=='1'):
                        weights[self.labelIndexes[label]][self.featureIndexes[feature]]+=1.0
                    
                    elif label.strip() == 'O' and datum.label == label and datum.features[feature]=='0':
                        weights[self.labelIndexes[label]][self.featureIndexes[feature]]+=1.0
                    else:
                        continue
                            
                        
        return weights
    
    
    def getProb(self, datum):
        counts = {}
        
        for label in self.labels:
            if label == "O":
                counts[label] = 0
                for feature in self.features:
                    if feature == '0':
                        counts[label] += self.lambdas[self.featureIndexes[feature]]
            elif label != "O":
                counts[label] = 0
                for feature in datum.features:
                    if feature == '1':
                        counts[label] += self.lambdas[self.featureIndexes[feature]]
                        
        exponents = {}
        for key in counts.keys():
            exponents[key] = np.exp(counts[key])
        
        return exponents                
    
    
    def predtictedCount(self,trainFeatures):
        for feature in self.features:
            for datum in data:
                sum = 0
                sumOfExponents = 0
                exponents = self.getProb(datum)
                for labExp in exponents.keys():
                    sumOfExponents += exponents[labExp]
                     
                for label in self.labels:
                    prob = exponents[label]/sumOfExponents
                    
                
max = MaximumEntropy()
data = max.readTrainFile('trainWithFeatures1.csv')
weight = max.empericalCount(data)
print(weight)
