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
        
        print("")    
        feat = list(data[0].features.keys())
        self.features = feat
        print("feat", self.featureIndexes)
        for i in range(0,len(feat)):
            #print("asd",feat)
            self.featureIndexes[feat[i]] = i  
        
        return data  
            
            #datum.features = {'g1':row['g1'],'g2':row['g2'],'g3':row['g3'],'g4':row['g4'],'g5':row['g5'],'g6':row['g6'],'g7':row['g7'],}

    def empericalCount(self,trainFeatures):
        
        weights = np.zeros(shape=(len(self.labels),len(trainFeatures[0].features)))
        for datum in trainFeatures:
            #print(datum.word)
            for label in self.labels:
                #print("las",label)
                #print(self.features)
                for feature in self.features:
                    #print("1",label,label=='O')
                    #print("2",datum.label,datum.label==label)
                    #print("3",datum.features[feature],datum.features[feature]==1)
                    #print("4",datum.features[feature]=='0')
                    #print("l",label,feature)
                    if (label.strip() != 'O') and (datum.label == label) and (datum.features[feature]=='1'):
                        #print("asdf",datum.features[feature])
                        weights[self.labelIndexes[label]][self.featureIndexes[feature]]+=1.0
                    
                    elif label.strip() == 'O' and datum.label == label and datum.features[feature]=='0':
                        #print("asdf",datum.features[feature])
                        weights[self.labelIndexes[label]][self.featureIndexes[feature]]+=1.0
                    else:
                        continue
                            
                        
        return weights
    
    
    def predtictedCount(self,trainFeatures):
        
        
        print("")
                
max = MaximumEntropy()
data = max.readTrainFile('trainWithFeatures1.csv')
weight = max.empericalCount(data)
print(weight)
