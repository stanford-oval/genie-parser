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
Created on Mar 1, 2018

@author: gcampagn
'''

import tornado.web

import tensorflow as tf


class BaseAdminHandler(tornado.web.RequestHandler):
    def check_authenticated(self):
        if not self.application.config.admin_token:
            raise tornado.web.HTTPError(500, "Admin token is not configured, cannot perform admin operations")
        
        admin_token = self.get_query_argument('admin_token')
        if admin_token != self.application.config.admin_token:
            raise tornado.web.HTTPError(401, "Unauthorized")


class ReloadHandler(BaseAdminHandler):
    def post(self, locale='en-US', model_tag=None, **kw):
        self.check_authenticated()
        language = self.application.get_language(locale, model_tag)
        self.application.reload_language(language.language_tag, language.model_tag)
        self.write(dict(result='ok'))
        self.finish()


class ExactMatcherReload(BaseAdminHandler):
    def post(self, locale='en-US', model_tag=None, **kw):
        self.check_authenticated()
        language = self.application.get_language(locale, model_tag)
        tf.logging.info('Reloading exact matches for %s', language.tag)
        language.exact.load()
        self.write(dict(result='ok'))
        self.finish()