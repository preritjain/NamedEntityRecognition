'''
Created on 29-Apr-2016

@author: prerit
'''
import base64
import json
from core.FeatureFactory import FeatureFactory



def readFeatures(filename):
    with open(filename) as json_file:
        json_data = json.load(json_file)
        print(json_data)
        
def runMEMM(trainFile,testFile):
    featureFactory=FeatureFactory() 
    trainData = FeatureFactory.readData(featureFactory, trainFile)
    print(trainData[1].word)
    testData = readFeatures(testFile)
    


if __name__ == '__main__':
    trainFile = 'trainWithFeatures.json'
    testFile = 'testWithFeatures.json'
    testData = runMEMM(trainFile,testFile)
    