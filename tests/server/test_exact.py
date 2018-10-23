# Copyright 2018 The Board of Trustees of the Leland Stanford Junior University
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

from luinet.server.exact import ExactMatcher

def test_exact_basic():
    matcher = ExactMatcher(None, 'en', 'default')
    
    matcher.add('get xkcd', 'now => @com.xkcd.get => notify')
    matcher.add('post on twitter', 'now => @com.twitter.post')
    matcher.add('post on twitter saying foo', 'now => @com.twitter.post param:status:String = " foo "')
    
    assert matcher.get('post on twitter') == ('now => @com.twitter.post'.split(' '))
    assert matcher.get('post on twitter saying foo') == ('now => @com.twitter.post param:status:String = " foo "'.split(' '))
    
    assert matcher.get('post on facebook') == None
    assert matcher.get('post on twitter saying lol') == None
    assert matcher.get('post on') == None