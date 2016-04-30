import json, sys
import base64
#import Datum
import nltk
from core.Datum import Datum
'''
Created on 29-Apr-2016

@author: prerit
'''

def getShape( word):
    s=""
    #s.center(len(word))        
    if len(word) >= 4:
        s='X' + s if word[0].isupper() else 'x' + s
        s='X' + s if word[1].isupper() else 'x' + s
        temp = ""
        for i in range(2,len(word)-2):
            if word[i].isupper():
                if temp.find('X')<0:
                    temp+= 'X'
            
            elif word[i].islower():
                if temp.find('x') < 0:
                    temp+='x'
                    
            elif word[i].isdigit():
                if temp.find('d') < 0:
                    temp+='d'
                    
            else:
                temp+= word[i]
        
        s = s  + temp    
        s+='X' if word[-1].isupper() else 'x'
        s+='X' if word[-2].isupper() else 'x'
    
    else:
        for i in range(0,len(word)):
            if word[i].isupper():
                s+='X'
            if word[i].islower():
                s+='X'
            if word[i].isdigit():
                s+='d'
            else:
                 s+= word[i]    
            
    s = ''.join(s.split())    
    return s
        
class FeatureFactory:
    """
    Add any necessary initialization steps for your features here
    Using this constructor is optional. Depending on your
    features, you may not need to intialize anything.
    """
    def __init__(self):
        pass

    
    """
    Words is a list of the words in the entire corpus, previousLabel is the label
    for position-1 (or O if it's the start of a new sentence), and position
    is the word you are adding features for. PreviousLabel must be the
    only label that is visible to this method. 
    """
    
    def computeFeatures(self, words, previousLabel, position):
        features = []
        currentWord = words[position]

        """ Baseline Features """
        features.append("word=" + currentWord)
        features.append("prevLabel=" + previousLabel)
        features.append("word=" + currentWord + ", prevLabel=" + previousLabel)
        features.append("start_case="+str(currentWord[0].isupper()))
        features.append("shape=" + getShape(currentWord))
        if(words[position-1]=='.'):
            features.append("newSent=True")
        tag = nltk.pos_tag(nltk.word_tokenize(currentWord))[0][1]
        features.append("tag=" + tag)
        '''
        features.append("shape"+getShape(currentWord))
        
        desiredTags = ['NN','NNP']
        
        if tag in desiredTags:
            features.append("tag=yes")
        else:
            features.append("tag=no")
        '''    
        return features        
    """
        Warning: If you encounter "line search failure" error when
        running the program, considering putting the baseline features
    back. It occurs when the features are too sparse. Once you have
        added enough features, take out the features that you don't need. 
    """


    """ TODO: Add your features here """

        

    """ Do not modify this method """
    def readData(self, filename):
        data = [] 
        
        for line in open(filename, 'r'):
            line_split = line.split()
            # remove emtpy lines
            if len(line_split) < 2:
                continue
            word = line_split[0]
            label = line_split[1]

            datum = Datum(word, label)
            data.append(datum)

        return data

    """ Do not modify this method """
    def readTestData(self, ch_aux):
        data = [] 
        
        for line in ch_aux.splitlines():
            line_split = line.split()
            # remove emtpy lines
            if len(line_split) < 2:
                continue
            word = line_split[0]
            label = line_split[1]

            datum = Datum(word, label)
            data.append(datum)

        return data


    """ Do not modify this method """
    def setFeaturesTrain(self, data):
        newData = []
        words = []

        for datum in data:
            words.append(datum.word)

        ## This is so that the feature factory code doesn't
        ## accidentally use the true label info
        previousLabel = "O"
        for i in range(0, len(data)):
            datum = data[i]

            newDatum = Datum(datum.word, datum.label)
            newDatum.features = self.computeFeatures(words, previousLabel, i)
            newDatum.previousLabel = previousLabel
            newData.append(newDatum)

            previousLabel = datum.label

        return newData

    """
    Compute the features for all possible previous labels
    for Viterbi algorithm. Do not modify this method
    """
    def setFeaturesTest(self, data):
        newData = []
        words = []
        labels = []
        labelIndex = {}

        for datum in data:
            words.append(datum.word)
            if not datum.label not in labelIndex.keys():
                labelIndex[datum.label] = len(labels)
                labels.append(datum.label)
        
        ## This is so that the feature factory code doesn't
        ## accidentally use the true label info
        for i in range(0, len(data)):
            datum = data[i]

            if i == 0:
                previousLabel = "O"
                datum.features = self.computeFeatures(words, previousLabel, i)

                newDatum = Datum(datum.word, datum.label)
                newDatum.features = self.computeFeatures(words, previousLabel, i)
                newDatum.previousLabel = previousLabel
                newData.append(newDatum)
            else:
                for previousLabel in labels:
                    datum.features = self.computeFeatures(words, previousLabel, i)

                    newDatum = Datum(datum.word, datum.label)
                    newDatum.features = self.computeFeatures(words, previousLabel, i)
                    newDatum.previousLabel = previousLabel
                    newData.append(newDatum)

        return newData

    """
    write words, labels, and features into a json file
    Do not modify this method
    """
    def writeData(self, data, filename):
        outFile = open(filename + '.json', 'w')
        for i in range(0, len(data)):
            datum = data[i]
            jsonObj = {}
            jsonObj['_label'] = datum.label
            jsonObj['_word']= datum.word
            jsonObj['_prevLabel'] = datum.previousLabel

            featureObj = {}
            features = datum.features
            for j in range(0, len(features)):
                feature = features[j]
                featureObj['_'+feature] = feature
            jsonObj['_features'] = featureObj
            
            outFile.write(json.dumps(jsonObj) + '\n')
            
        outFile.close()
    