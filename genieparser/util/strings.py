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
Created on Oct 23, 2018

@author: gcampagn
'''

import re


def clean(name):
    """Normalize argument names into English words.
    
    Removes the "v_" prefix, converts camelCase to multiple words, and underscores
    to spaces.
    """ 
    if name.startswith('v_'):
        name = name[len('v_'):]
    return re.sub('([^A-Z])([A-Z])', '$1 $2', re.sub('_', ' ', name)).lower()


def tokenize(name):
    return re.split(r'\s+|[,\.\"\'!\?]', re.sub('[()]', '', name.lower()))


def find_substring(sequence, substring):
    for i in range(len(sequence)-len(substring)+1):
        found = True
        for j in range(0, len(substring)):
            if sequence[i+j] != substring[j]:
                found = False
                break
        if found:
            return i
    return -1


def find_span(input_sentence, span):
    # empty strings have their own special token "",
    # they should not appear here
    assert len(span) > 0

    input_position = find_substring(input_sentence, span)

    if input_position < 0:
        raise ValueError("Cannot find span \"%s\" in \"%s\"" % (span, input_sentence))

    # NOTE: the boundaries are inclusive (so that we always point
    # inside the span)
    # NOTE 2: the input_position cannot be zero, because
    # the zero-th element in input_sentence is <s>
    # this is important because zero is used as padding/mask value
    return input_position, input_position + len(span)-1