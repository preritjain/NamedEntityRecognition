'''
Created on 03-May-2016

@author: prerit
'''

class Datum:
    '''
    classdocs
    '''


    def __init__(self, word, label):
        self.word = word
        self.label = label
        self.guessLabel = ''
        self.previousLabel = ''
        self.features = {}
        '''
        Constructor
        '''
        