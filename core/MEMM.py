'''
Created on 29-Apr-2016

@author: prerit
'''
import numpy as np
import base64
import math
import json
from core.Datum import Datum
from core.FeatureFactory import FeatureFactory
#from py2neo.core import LabelSet
#from core.NER import trainData




def readFeatures(filename):
    with open(filename) as json_file:
        json_data = json.load(json_file)
        #print(json_data)
        
def runMEMM(trainFile,testFile):
    
    trainData = read(trainFile)
    testDataWithMultiplePrevLabels = read(testFile)
    labelSet = []
    labelIndexes = {}
    featureSet = []
    featureIndexes = {}
    
    for datum in trainData:
        
        if labelIndexes.get(datum.label) == None:
            lIndex = len(labelSet)
            labelSet.append(datum.label)
            labelIndexes[datum.label] = lIndex
        
        for feature in datum.features:
            if featureIndexes.get(feature) == None:
                fIndex = len(featureSet)
                featureSet.append(feature)
                featureIndexes[feature] = fIndex
    
    testData = []
    testData.append(testDataWithMultiplePrevLabels[0])
    
    for i in range(1,len(testDataWithMultiplePrevLabels)-1, len(labelSet)):
        testData.append(testDataWithMultiplePrevLabels[i])
    calculateLiklihood(trainData,labelSet,featureSet,featureIndexes)
    print(testDataWithMultiplePrevLabels[0].features)
    print(labelSet)
    print(featureSet[0:10])
    #print(featureIndexes)
'''
logsum sums over all classes(exp(sum(lambda(j) * fj(ci,d)))
'''    
def logsum(scores):
        maxId = 0
        max=scores[0]
        for t in range(0,len(scores)):
            if scores[t] > max:
                maxId = t
                max = scores[t]
        cutoff = max - 30
        intermediate = 0.0
        haveTerms = False
        
        
        for i in range(0,len(scores)):
            if i != maxId and scores[i] > cutoff:
                haveTerms = True
                intermediate += math.exp(scores[i]-max)
        if haveTerms:
            return max + math.log(1+intermediate)
        else:
            return max
        
def logExp(scores):
    pass
                    

def likelihood(trainData,labelSet,featureSet, featureIndexes):                     
def calculateLiklihood(trainData,labelSet,featureSet, featureIndexes):
    value = 0.0
    #features = 
    features = np.zeros(len(labelSet)*len(featureSet))
    derivatives = np.zeros(shape=(len(labelSet),int(len(features)/len(labelSet))))
    weights = features.reshape(len(labelSet),int(len(features)/len(labelSet)))
    print("weights")
    print(weights)
    
    for datum in trainData:
        scores = list(np.zeros(len(labelSet)))
        print(datum.label)
        for feats in datum.features:
            if featureIndexes.get(feats) != None:
                f = featureIndexes[feats]
                for i in range(0,len(labelSet)):
                    scores[i]+= weights[i][f]
                     
                    
        z = logsum(scores) 
        for i in range(0,len(scores)):
            prob = math.exp(scores[i]-z)
            for feats in datum.features:
                pass    
                    

def read(fileName):
    data = []
    with open(fileName,"r") as inputFile:
        line = inputFile.readline()
        while line != "":
            jsonLine = json.loads(line)
            #print(jsonLine)
            
            word = jsonLine["_word"]
            label = jsonLine["_label"]
            prevLabel = jsonLine["_prevLabel"]
            features = []
            featuresDict = jsonLine["_features"]
            for feature in featuresDict.keys():
                features.append(featuresDict[feature])
            
            datum = Datum(word,label)
            datum.features = features
            datum.previousLabel = prevLabel
            
            data.append(datum)
            line = inputFile.readline()

    return data
        

def MEMM():
    trainFile = 'trainWithFeatures.json'
    testFile = 'testWithFeatures.json'
    testData = runMEMM(trainFile,testFile)
    
MEMM()
    