#!/usr/bin/python3

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
