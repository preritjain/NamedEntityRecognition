'''
Created on 03-May-2016

@author: prerit
'''
import re
import csv
import nltk
import json, sys
from maxent.Datum import Datum 
class FeatureFactory:
    
    '''
    classdocs
    '''


    def __init__(self):
        self.names = frozenset(nltk.corpus.names.words())
        self.midname = re.compile("^[A-Z]\.$")
        '''
        Constructor
        '''
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
            #print("Word: ", datum.word)
            #print("Label: ", datum.label)
            data.append(datum)

        return data
    
    """
    Fi features for a word   
    """
    
    def g1_case(self,word):
        if word[0].isupper():
            return 1
        else: 
            return 0
    
    
    def g2_current_pos(self,word):
        desiredTags= ['NNP','NNPS','NN']
        tag = nltk.pos_tag(nltk.word_tokenize(word))[0][1]
        if tag in desiredTags:
            return 1
        else:
            return 0
        
    def g3_gazeteers(self,word):
        
        if word.endswith("'s"):
            if word[:-2] in self.names:
                return 1
        
        if word in self.names:
            return 1
        else:
            return 0    
        
        
    def g4_mister(self,word,previousWord):
        mister = frozenset(['mr',"ms","ms.","mss","mr.","dr","dr.","de","fon"])
        if previousWord.lower() in mister:
            return 1
        else:
            return 0
        
    def g5_midNames(self,word):
        if self.midname.match(word):
            return 1
        else:
            return 0
        
    def g6_prevPos_and_currentCase(self,word,previousWord):
        if self.g2_current_pos(previousWord):
            if self.g1_case(word):
                return 1
        return 0
    
    def g7_secondName(self,word,prevLabel):
        if prevLabel == 'PERSON':
            if self.g1_case(word):
                return 1    
        return 0
    
    
    """
    Words is a list of the words in the entire corpus, previousLabel is the label
    for position-1 (or O if it's the start of a new sentence), and position
    is the word you are adding features for. PreviousLabel must be the
    only label that is visible to this method. DONE
    """
    
    def computeFeatures(self, words, previousLabel, position):
        features = {}
        currentWord = words[position]
        previousWord = words[position-1] or "=NOT-A-WORD="
        features["g1"]=self.g1_case(currentWord)
        features["g2"] = self.g2_current_pos(currentWord)
        features["g3"] = self.g3_gazeteers(currentWord)
        features["g4"] = self.g4_mister(currentWord,previousWord)
        features["g5"]=self.g5_midNames(currentWord)
        features["g6"] = self.g6_prevPos_and_currentCase(currentWord,previousWord)
        features["g7"] = self.g7_secondName(currentWord,previousLabel)
        
        #print("came in compute")
    
        return features
        
        
        
    def setFeaturesTrain(self, data):
        newData = []
        words = []
        print("length", len(data))
        for datum in data:
            words.append(datum.word)

        ## This is so that the feature factory code doesn't
        ## accidentally use the true label info
        count=0
        previousLabel = "O"
        for i in range(0, len(data)):
            
            datum = data[i]

            newDatum = Datum(datum.word, datum.label)
            newDatum.features = self.computeFeatures(words, previousLabel, i)
            newDatum.previousLabel = previousLabel
            newData.append(newDatum)

            previousLabel = datum.label
            print(count)
            count= count+1
        print("return data")
        return newData
    
    def writeData(self,data,filename):  
        #outFile = open(filename + '.csv', 'w')
        filename = filename + ".csv"
        fnames = ["word"]
        for i in range(1,len(data[0].features)+1):
            k= 'g' + str(i)
            fnames.append(k)
        fnames.append('label')
        writer = csv.DictWriter(open(filename,'a'), fieldnames = fnames)
        writer.writeheader()
            
        for datum in data:
            row = {"word":datum.word,'label':datum.label}
            for i in range(1,len(data[0].features)+1):
                k = 'g' + str(i)
                row[k] = datum.features[k]
            writer.writerow(row)
    
    
    


         