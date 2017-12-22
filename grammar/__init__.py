# Copyright 2017 Giovanni Campagna <gcampagn@cs.stanford.edu>
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

from .abstract import AbstractGrammar
from .simple import SimpleGrammar
from .thingtalk import ThingtalkGrammar
from .new_thingtalk import NewThingTalkGrammar

def create_grammar(grammar_type, input_file):
    if grammar_type == "tt":
        return ThingtalkGrammar(input_file)
    elif grammar_type == 'new-tt':
        return NewThingTalkGrammar(input_file)
    elif grammar_type == "simple":
        return SimpleGrammar(input_file)
    else:
        raise ValueError("Invalid grammar %s" % (grammar_type,))