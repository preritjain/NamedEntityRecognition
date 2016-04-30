'''
Created on 29-Apr-2016

@author: prerit
'''
import base64
from core.FeatureFactory import FeatureFactory

if __name__ == '__main__':
    featureFactory = FeatureFactory()
    trainFile = '../data/train'
    testFile = '../data/dev'
    # read the train and test data
    trainData = featureFactory.readData(trainFile)
    testData = featureFactory.readData(testFile)

    # add the features
    trainDataWithFeatures = featureFactory.setFeaturesTrain(trainData);
    testDataWithFeatures = featureFactory.setFeaturesTest(testData);

    # write the updated data into JSON files
    featureFactory.writeData(trainDataWithFeatures, 'trainWithFeatures');
    featureFactory.writeData(testDataWithFeatures, 'testWithFeatures');