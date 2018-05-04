'''
Created on Apr 24, 2018

@author: gcampagn
'''

from .shift_reduce_grammar import ShiftReduceGrammar

from collections import OrderedDict

class IFTTTGrammar(ShiftReduceGrammar):
    '''
    The grammar of ThingTalk
    '''
    
    def __init__(self, filename=None, **kw):
        super().__init__(**kw)
        self.entities = []
        
        GRAMMAR = OrderedDict({
            '$input': [('$trigger', '=>', '$action')],
            '$channel': [],
            '$trigger': [('$channel', '$trigger_function')],
            '$trigger_function': [],
            '$action': [('$channel', '$action_function')],
            '$action_function': [],
        })
    
        self.allfunctions = set()
        with open(filename, 'r') as fp:
            for line in fp:
                where, what = line.strip().split(' ')
                if where == 'channel':
                    GRAMMAR['$channel'].append((what,))
                #elif where == 'param':
                #    GRAMMAR['$param_name'].append((what,))
                #elif where == 'value':
                #    GRAMMAR['$param_value'].append((what,))
                else:
                    GRAMMAR['$' + where + '_function'].append((what,))
                    self.allfunctions.add(what)

        self.num_functions = len(GRAMMAR['$trigger_function']) + len(GRAMMAR['$action_function'])
        self.tokens += self.construct_parser(GRAMMAR)

        print('num channels', len(GRAMMAR['$channel']))
        print('num functions', self.num_functions)
        print('num triggers', len(GRAMMAR['$trigger_function']))
        print('num actions', len(GRAMMAR['$action_function']))
        print('num other', len(self.tokens) - self.num_functions - self.num_control_tokens)
        
        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i
