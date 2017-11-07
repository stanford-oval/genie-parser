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

    output = dict()
    with urllib.request.urlopen(thingpedia_url + '/api/snapshot/' + str(snapshot) + '?meta=1', context=ssl_context) as res:
        output['devices'] = json.load(res)['data']

    with urllib.request.urlopen(thingpedia_url + '/api/entities?snapshot=' + str(snapshot), context=ssl_context) as res:
        output['entities'] = json.load(res)['data']
    
    json.dump(output, sys.stdout, indent=2)

main()
