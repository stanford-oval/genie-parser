'''
Created on Aug 2, 2017

@author: gcampagn
'''

import tornado.web
import json

class LearnHandler(tornado.web.RequestHandler):
    
    def get(self):
        query = self.get_query_argument("q")
        locale = self.get_query_argument("locale", default="en-US")
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