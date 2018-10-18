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
Created on Jul 26, 2018

@author: gcampagn, mehrad
'''

from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_tiny, transformer_base
from tensor2tensor.models.lstm import lstm_luong_attention

def luinet_extra_hparams(hp):
    hp.eval_run_autoregressive = True
    hp.add_hparam("grammar_direction", "bottomup")
    hp.add_hparam("use_margin_loss", False)
    hp.add_hparam("train_input_embeddings", False)
    hp.add_hparam("pointer_layer", "attentive")
    hp.set_hparam("num_hidden_layers", 2)
    hp.set_hparam("hidden_size", 128)
    hp.set_hparam("filter_size", 512)
    hp.set_hparam("num_heads", 4)

@registry.register_hparams
def transformer_luinet():
    # Start with the base set
    # default is transformer_tiny
    hp = transformer_base()
    luinet_extra_hparams(hp)
    return hp

@registry.register_hparams
def lstm_luinet():
    hp = lstm_luong_attention()
    luinet_extra_hparams(hp)
    return hp
