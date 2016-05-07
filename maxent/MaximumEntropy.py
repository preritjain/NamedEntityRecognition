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
    
    
    def empericalCountDatum(self,datum):
        weights = np.zeros(shape=(len(self.labels),len(datum.features)))
        for label in self.labels:
                for feature in self.features:
                    if (label.strip() != 'O') and (datum.label == label) and (datum.features[feature]=='1'):
                        weights[self.labelIndexes[label]][self.featureIndexes[feature]]+=1.0
                    
                    elif label.strip() == 'O' and datum.label == label and datum.features[feature]=='0':
                        weights[self.labelIndexes[label]][self.featureIndexes[feature]]+=1.0
                    else:
                        continue
                            
        #print("there",weights)                
        return weights
    
    
    
    def predictedCount(self,trainFeatures):
        #for feature in self.features:
        print("lambda",self.lambdas)
        predCount = np.zeros(shape=(len(self.labels),len(trainFeatures[0].features)))
        for datum in data:
            empCount = self.empericalCountDatum(datum)
            count = empCount
            #print("features",datum.features)
            
            sum = 0
            sumOfExponents = 0
            exponents = self.getProb(datum)
            #print("exp",exponents)
            for labExp in exponents.keys():
                sumOfExponents += exponents[labExp]
                #print("feat here",datum.features)     
            for label in self.labels:
                #print("sum",sumOfExponents)
                prob = exponents[label]/sumOfExponents
                #print("prob",prob)
                for feature in datum.features:
                    count[self.labelIndexes[label]][self.featureIndexes[feature]]*=prob 
                    predCount[self.labelIndexes[label]][self.featureIndexes[feature]]+=count[self.labelIndexes[label]][self.featureIndexes[feature]]               
        #print("here",predCount)
        print(prob)
        return predCount
        
        
        
    def getProb(self, datum):
        counts = {}
        #print("features ",datum.features)
        #print("label ",datum.label)
        
        for label in self.labels:
           
            if label == "O":
                counts[label] = 0
                for feature in self.features:
                    #print("di",feature)
                    if datum.features[feature] == '0' and datum.label == label:
                        #print("did")
                        counts[label] += self.lambdas[self.featureIndexes[feature]]
            elif label != "O":
                counts[label] = 0
                for feature in self.features:
                    #print("didi")
                    if datum.features[feature] == '1' and datum.label == label:
                        #print("didii")
                        counts[label] += self.lambdas[self.featureIndexes[feature]]
                        
        exponents = {}
        #print("counts",counts)
        for key in counts.keys():
            exponents[key] = np.exp(counts[key])
        #print("exponents",exponents)
        return exponents                
    
    """
    def predtictedCount(self,trainFeatures):
        for feature in self.features:
            for datum in data:
                empCount = self.empericalCountDatum(datum)
                sum = 0
                sumOfExponents = 0
                exponents = self.getProb(datum)
                for labExp in exponents.keys():
                    sumOfExponents += exponents[labExp]
                     
                for label in self.labels:
                    prob = exponents[label]/sumOfExponents
                    
    """                
    
    def computeCost(self,data,empericalCount):
        
        
        predC = max.predictedCount(data)
        predC1D = predC.sum(axis=0)
        cost  = empericalCount - predC1D
        print("cost",cost)
        return cost
    
    def evaluateDatum(self,datum):
        
        
        empCount = self.empericalCountDatum(datum)
        count = empCount
        #print("features",datum.features)
        
        sum = 0
        sumOfExponents = 0
        exponents = self.getProb(datum)
        #print("exp",exponents)
        for labExp in exponents.keys():
            sumOfExponents += exponents[labExp]
            #print("feat here",datum.features)     
        for label in self.labels:
            #print("sum",sumOfExponents)
            prob = exponents[label]/sumOfExponents
            #print("prob",prob)
            for feature in datum.features:
                count[self.labelIndexes[label]][self.featureIndexes[feature]]*=prob 
        
        pred = count.sum(axis = 1)
        print("prediction probabilities: ",count)
        print("actual label, ",datum.label)
        
    def getProbTest(self,datum):
        counts = {}
        #print("features ",datum.features)
        #print("label ",datum.label)
        
        for label in self.labels:
           
            if label == "O":
                counts[label] = 0
                for feature in self.features:
                    #print("di",feature)
                    if datum.features[feature] == '0':
                        #print("did")
                        counts[label] += self.lambdas[self.featureIndexes[feature]]
            elif label != "O":
                counts[label] = 0
                for feature in self.features:
                    #print("didi")
                    if datum.features[feature] == '1':
                        #print("didii")
                        counts[label] += self.lambdas[self.featureIndexes[feature]]
                        
        exponents = {}
        #print("counts",counts)
        for key in counts.keys():
            exponents[key] = np.exp(counts[key])
        #print("exponents",exponents)
        return exponents     
        
    def evaluate(self,data):
        f = open("out.txt", mode='w')
        true = 0
        truePersons = 0
        totalPersons = 0
        total = 0
        for datum in data:
            total+=1
            #empCount = self.empericalCountDatum(datum)
            #count = empCount
            #print("features",datum.features)
            prob = {}
            sum = 0
            if datum.label == 'PERSON':
                totalPersons+=1
            sumOfExponents = 0
            exponents = self.getProbTest(datum)
            #print("exp",exponents)
            for labExp in exponents.keys():
                sumOfExponents += exponents[labExp]
                #print("feat here",datum.features)     
            for label in self.labels:
                #print("sum",sumOfExponents)
                prob[label] = exponents[label]/sumOfExponents
                
                #print("prob",prob)
                #for feature in datum.features:
                    #count[self.labelIndexes[label]][self.featureIndexes[feature]]*=prob 
            
            #pred = count.sum(axis = 1)
            #print("prediction probabilities: ",pred)
            
            max = 0
            guess_label = 'O'
            for lab in prob.keys():
                if prob[lab] > max:
                    max = prob[lab]
                    guess_label = lab
                    
            datum.guessLabel = guess_label          
            if guess_label == datum.label:
                true+=1
            if guess_label == datum.label and guess_label == 'PERSON':
                truePersons+=1
                        
            #print(datum.word,datum.label,guess_label)
            f.write(datum.word +" " + datum.label + " " + guess_label + str(prob['O']) + " "+ str(prob['PERSON']) +  "\n" )
            """
            if (pred[0]>pred[1]):
                lab = 'O'
                if(datum.label == 'O'):
                    true+=1
                
                print(datum.word,datum.label,'O')
            elif (pred[1]>pred[0]):
                lab = 'PERSON'
                if(datum.label == 'PERSON'):
                    true+=1
                
                print(datum.word,datum.label,'PERSON')   
            print(datum.word,datum.label,lab) 
            """  
        print("true classifications=  ", true,"/",total)
        print("truePerSOns", truePersons, "/",totalPersons)
        f.write(str(truePersons) +  "/" + str(totalPersons))
        f.close()
                    
    def gradientDescent(self,data,empC1D):
        iter = 0
        print("iter",iter)
        
        dnBYdm = self.computeCost(data,empC1D)
        #print("a",self.lambdas[1])
        dnBYdm = -1*dnBYdm
        print("b",dnBYdm)
        temp = self.lambdas
        numIter = 100
        l_rate = 0.002
        for i in range(0,numIter):
            for j in range(0,len(self.lambdas)):
                temp[j] = (self.lambdas[j]) - (l_rate)*(dnBYdm[j])
            
            self.lambdas = temp
            print("temp",temp)
       
            print("dnBYdm",dnBYdm)
            dnBYdm = -1*(self.computeCost(data,empC1D))
        iter +=1
            
            
                
max = MaximumEntropy()
data = max.readTrainFile('trainWithFeatures1.csv')
#data = max.readTrainFile('testWithFeatures.csv')
empC = max.empericalCount(data)
empC1D = empC.sum(axis=0)
max.lambdas = np.zeros(len(max.features))
max.lambdas = np.ones(len(max.features))
#max.gradientDescent(data,empC1D)
#empC = max.empericalCount(data)
#empC1D = empC.sum(axis=0)
#print("empCount",empC)
#print("sum emp count", empC1D)
#predC = max.predictedCount(data)
#predC1D = predC.sum(axis=0)
max.lambdas = np.array([4.63288814  ,2.3448111  , 2.31470926 , 3.37571473 , 2.54351436,  2.51013194, 2.13468754])
max.evaluate(data)



'''
lambda [ 4.63288814  2.3448111   2.31470926  3.37571473  2.54351436  2.51013194
  2.13468754]
  '''

  


#print("predCOunt",predC1D)

#print("lambda",max.lambdas)
#print("dnBYdM",dnBYdm)
