'''
Created on Nov 6, 2017

@author: gcampagn
'''

class ExactMatcher():
    def __init__(self, database, language):
        self._database = database
        self._language = language
        
        self._dict = dict() 
    
    def load(self):
        n = 0
        for row in self._database.execute("select utterance,target_json from example_utterances where language =  %(language)s and type = 'online'",
                                          language=self._language):
            self._dict[row['utterance']] = row['target_json']
            n += 1
        print('Loaded %d exact matches for language %s' % (n, self._language))
            
    def add(self, utterance, target_json):
        self._dict[utterance] = target_json
        
    def get(self, utterance):
        return self._dict.get(utterance, None)