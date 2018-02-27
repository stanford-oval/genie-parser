# Copyright 2017-2018 The Board of Trustees of the Leland Stanford Junior University
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
Created on Aug 2, 2017

@author: gcampagn
'''

import tornado.web
import json

class LearnHandler(tornado.web.RequestHandler):
    def get(self, *kw):
        query = self.get_query_argument("q")
        locale = kw.get('locale', None) or self.get_query_argument("locale", default="en-US")
        language = self.application.get_language(locale)
        target_json = self.get_query_argument("target")
        store = self.get_query_argument("store", "automatic")
        
        try:
            json.loads(target_json)
        except Exception:
            raise tornado.web.HTTPError(400, "Invalid JSON")
        
        if store == 'no':
            self.finish(dict(result="Learnt successfully"))
            return
        
        if not store in ('automatic', 'online'):
            raise tornado.web.HTTPError(400, "Invalid store parameter")
        
        if not self.application.database:
            raise tornado.web.HTTPError(500, "Server not configured for online learning")
        self.application.database.execute("insert into example_utterances (is_base, language, type, utterance, target_json, click_count) " +
                                          "values (0, %(language)s, %(type)s, %(utterance)s, %(target_json)s, -1)",
                                          language=language.tag,
                                          utterance=query,
                                          type=store,
                                          target_json=target_json)
        #if language.exact and store == 'online':
        #    language.exact.add(query, target_json)