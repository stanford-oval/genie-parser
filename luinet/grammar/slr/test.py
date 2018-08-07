# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
#
# Author: Giovanni Campagna <gcampagn@cs.stanford.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>. 
'''
Created on Nov 30, 2017

@author: gcampagn
'''

from .generator import SLRParserGenerator


TEST_GRAMMAR = {
'$prog':    [('$command',),
             ('$rule',)],

'$rule':    [('$stream', '$action'),
             ('$stream', 'notify')],
'$command': [('$table', 'notify'),
             ('$table', '$action')],
'$table':   [('$get',),
             ('$table', 'filter', '$filter')],
'$stream':  [('monitor', '$table')],
'$get':     [('$get', '$ip'),
             ('GET',)],
'$action':  [('$action', '$ip'),
             ('DO',)],
'$ip':      [('PARAM', '$number'),
             ('PARAM', '$string')],
'$number':  [('num0',),
             ('num1',)],
'$string':  [('qs0',),
             ('qs1',)],
'$filter':  [('PARAM', '==', '$number'),
             ('PARAM', '>', '$number'),
             ('PARAM', '<', '$number'),
             ('PARAM', '==', '$string'),
             ('PARAM', '=~', '$string')]
}
TEST_TERMINALS = {
    'PARAM': { 'param:number', 'param:text' },
    'GET': { 'xkcd.get_comic', 'thermostat.get_temp', 'twitter.search' },
    'DO': { 'twitter.post' }
}

# The grammar of nesting parenthesis
# this is context free but not regular
# (also not parsable with a Petri-net)
#
# The reduce sequence will be: the reduction of the inner most parenthesis pair,
# followed by the reduction for the next parenthesis, in order
#
# This has one important consequence: if the NN produces the sequence
# X Y Z* ...
# where X is reduction 4 or 5 (a or b), Y is reduction 0 or 1, and Z
# is 2 or 3, it will automatically produce a well-formed string in the language
# The NN is good at producing never ending sequences of the same thing (in
# fact, it tends to do that too much), so it should have no trouble with
# this language 
PARENTHESIS_GRAMMAR = {
'$S': [('(', '$V', ')'),
       ('[', '$V', ']'),
       ('(', '$S', ')'),
       ('[', '$S', ']')],
'$V': [('a',), ('b',)]
}


if __name__ == '__main__':
    if True:

        generator = SLRParserGenerator(TEST_GRAMMAR, start_symbol='$prog')
        #print("Action table:")
        #for i, actions in enumerate(generator.action_table):
        #    print(i, ":", actions)

        #print()          
        #print("Goto table:")
        #for i, next_states in enumerate(generator.goto_table):
        #    print(i, ":", next_states)
        
        parser = generator.build()
        
        def tokenize(seq):
            for token in seq:
                found = False
                for test_term, values in TEST_TERMINALS.items():
                    if token in values:
                        yield generator.dictionary[test_term], token
                        found = True
                        break
                if not found:
                    yield generator.dictionary[token], token

        print(parser.parse(tokenize(['monitor', 'thermostat.get_temp', 'twitter.post', 'param:text', 'qs0'])))
        
        def reconstruct(seq):
            for token_id, token in seq:
                yield token
        
        TEST_VECTORS = [
            ['monitor', 'thermostat.get_temp', 'twitter.post', 'param:text', 'qs0'],
            ['monitor', 'thermostat.get_temp', 'filter', 'param:number', '>', 'num0', 'notify'],
            ['thermostat.get_temp', 'filter', 'param:number', '>', 'num0', 'notify']
        ]
        
        for expected in TEST_VECTORS:
            assert expected == list(reconstruct(parser.reconstruct(parser.parse(tokenize(expected)))))
    else:
        generator = SLRParserGenerator(PARENTHESIS_GRAMMAR, start_symbol='$S')
        #print("Action table:")
        #for i, actions in enumerate(generator.action_table):
        #    print(i, ":", actions)
        
        #print()          
        #print("Goto table:")
        #for i, next_states in enumerate(generator.goto_table):
        #    print(i, ":", next_states)
        
        parser = generator.build()
        
        print(parser.parse(['(', '(', '(', 'a', ')', ')', ')']))
        print(parser.parse(['[', '[', '[', 'a', ']', ']', ']']))
        print(parser.parse(['(', '[', '(', 'b', ')', ']', ')']))