'''
Created on 29-Apr-2016

@author: prerit
'''
import base64
import json
from core.Datum import Datum
from core.FeatureFactory import FeatureFactory
from py2neo.core import LabelSet



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
    