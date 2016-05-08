import base64
from maxent.FeatureFactory import FeatureFactory
#from maxent.MaximumEntropy import MaximumEntropy
'''
Created on 03-May-2016

@author: prerit
'''

if __name__ == '__main__':
    trainFile = '../data/train'
    testFile = '../data/dev'
    #testFile = '../data/testInput.txt'
    featureFactory = FeatureFactory()
    # read the train and test data
    
    print("hello")
    trainData = featureFactory.readData(trainFile)
    testData = featureFactory.readData(testFile)
    print("here")
    trainDataWithFeatures = featureFactory.setFeaturesTrain(trainData)
    #testDataWithFeatures = featureFactory.setFeaturesTrain(testData);
    #maxent = MaximumEntropy()
    
    print('here too')
    #write the updated data into JSON files
    
    
    featureFactory.writeData(trainDataWithFeatures, 'trainWithFeatures3')
    #featureFactory.writeData(testDataWithFeatures, 'testWithFeatures');

