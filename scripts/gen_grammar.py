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

import MySQLdb
import MySQLdb.cursors
import sys
import json
import os

def main():
    conn = MySQLdb.connect(user='sempre', passwd=sys.argv[1],
                           db='thingengine',
                           use_unicode=True,
                           charset='utf8mb4',
                           host='thingengine.crqccvnuyu19.us-west-2.rds.amazonaws.com',
                           ssl=dict(ca=os.path.join(os.path.dirname(__file__), '../data/thingpedia-db-ca-bundle.pem')))
    cursor = conn.cursor(cursorclass=MySQLdb.cursors.DictCursor)
    cursor.execute("select kind, name, channel_type, argnames, types from device_schema ds, device_schema_channels dsc where ds.id = dsc.schema_id and dsc.version = ds.developer_version and kind_type <> 'global'")
    for row in cursor.fetchall():
        print(row['channel_type'], 'tt:' + row['kind'] + '.' + row['name'], end=' ')
        argnames = json.loads(row['argnames'])
        argtypes = json.loads(row['types'])
        for argname, argtype in zip(argnames, argtypes):
            print(argname, argtype, end=' ')
        print()
        
    cursor = conn.cursor(cursorclass=MySQLdb.cursors.DictCursor)
    cursor.execute("select kind from device_schema where kind_type <> 'global'")
    for row in cursor.fetchall():
        print('device', 'tt-device:' + row['kind'])
        
    cursor = conn.cursor(cursorclass=MySQLdb.cursors.DictCursor)
    cursor.execute("select id from entity_names where not is_well_known")
    for row in cursor.fetchall():
        print('entity', row['id'])
        
main()
