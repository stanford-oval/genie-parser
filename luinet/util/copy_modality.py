# Copyright 2018 Google LLC
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
Created on Jul 24, 2018

@author: gcampagn
'''

from tensor2tensor.utils import registry
from tensor2tensor.layers.modalities import IdentitySymbolModality


@registry.register_symbol_modality("copy")
class CopyModality(IdentitySymbolModality):
    '''
    Same as IdentitySymbolModality, but it's a subclass so
    MultiOutputTransformer can learn to recognize it.
    '''
        