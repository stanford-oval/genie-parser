#!/usr/bin/python3
#
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

import sys
import json
import os
import urllib.request
import ssl

def main():
    snapshot = int(sys.argv[1])
    thingpedia_url = os.getenv('THINGPEDIA_URL', 'https://thingpedia.stanford.edu/thingpedia')
    ssl_context = ssl.create_default_context()

    with urllib.request.urlopen(thingpedia_url + '/api/snapshot/' + str(snapshot), context=ssl_context) as res:
        data = json.load(res)['data']
        for device in data:
            if device['kind_type'] == 'global':
                continue
            print('device', 'tt-device:' + device['kind'])
            def do_type(dictionary, channel_type):
                for name, channel in dictionary.items():
                    print(channel_type, 'tt:' + device['kind'] + '.' + name, end=' ')
                    for argname, argtype, required, is_input in zip(channel['args'], channel['types'], channel['required'], channel['is_input']):
                        if is_input:
                            direction = 'in'
                        else:
                            direction = 'out'
                        print(argname, argtype, direction, end=' ')
                    print()
            do_type(device['triggers'], 'trigger')
            do_type(device['queries'], 'query')
            do_type(device['actions'], 'action')

    with urllib.request.urlopen(thingpedia_url + '/api/entities?snapshot=' + str(snapshot), context=ssl_context) as res:
        data = json.load(res)['data']
        for entity in data:
            if entity['is_well_known'] == 1:
                continue
            print('entity', entity['type'])

main()
