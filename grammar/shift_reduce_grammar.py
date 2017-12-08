'''
Created on Dec 8, 2017

@author: gcampagn
'''

import numpy as np

from .abstract import AbstractGrammar
from .slr import SLRParserGenerator

class ShiftReduceGrammar(AbstractGrammar):
    def __init__(self):
        super().__init__()
        
        self.tokens = ['<<PAD>>', '<<EOS>>', '<<GO>>']
        self.num_control_tokens = 3
        
        self._parser = None

    def construct_parser(self, grammar):
        generator = SLRParserGenerator(grammar, '$input')
        self._parser = generator.build()
        
        print('num rules', self._parser.num_rules)
        print('num states', self._parser.num_states)
        
    USE_SHIFT_REDUCE = True
    
    @property
    def output_size(self):
        if self.USE_SHIFT_REDUCE:
            # padding, eos, go, reduce 0 to n-1
            return self.num_control_tokens + self._parser.num_rules
        else:
            return len(self.tokens)

    def vectorize_program(self, program, max_length=60):
        if self.USE_SHIFT_REDUCE:
            if isinstance(program, str):
                program = program.split(' ')
            parsed = self._parser.parse(program)
            vector = np.zeros((max_length,), dtype=np.int32)
            i = 0
            for action, param in parsed:
                if action == 'shift':
                    continue
                if i >= max_length-1:
                    raise ValueError("Truncated parse of " + str(program) + " (needs " + str(len(parsed)) + " actions)")
                vector[i] = self.num_control_tokens + param
                i += 1
            vector[i] = self.end # eos
            i += 1
            
            return vector, i
        else:
            vector, length = super().vectorize_program(program, max_length)
            self.normalize_sequence(vector)
            return vector, length

    def reconstruct_program(self, sequence, ignore_errors=False):
        if self.USE_SHIFT_REDUCE:
            try:
                def gen_action(x):
                    if x <= self.end:
                        return ('accept', None)
                    else:
                        return ('reduce', x - self.num_control_tokens)
                return self._parser.reconstruct((gen_action(x) for x in sequence))
            except (KeyError, TypeError, IndexError, ValueError):
                if ignore_errors:
                    # the NN generated something that does not conform to the grammar,
                    # ignore it 
                    return []
                else:
                    raise
        else:
            return super().reconstruct_program(sequence)

    def parse(self, program):
        return self._parser.parse(program)

    def parse_all(self, fp):
        max_length = 0
        for line in fp.readlines():
            try:
                program = line.strip().split()
                parsed = self._parser.parse(program)
                reduces = [x for x in parsed if x[0] =='reduce']
                if len(reduces) > max_length:
                    max_length = len(reduces)
            except ValueError as e:
                print(' '.join(program))
                raise e