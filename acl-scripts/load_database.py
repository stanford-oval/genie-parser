#!/usr/bin/env python

import sys
import MySQLdb
import MySQLdb.cursors
import csv

conn = MySQLdb.connect(user='sempre', passwd=sys.argv[2],
                           db='thingengine',
                           host=sys.argv[1])
cursor = conn.cursor(cursorclass=MySQLdb.cursors.DictCursor)

filename = sys.argv[3]
if filename.split('.')[-1] == 'tsv':
    fp2 = open(filename, 'r')
    gen_examples = []
    for ex in fp2.read().splitlines():
        gen_examples.append((ex.split('\t')[0], ex.split('\t')[1]))
    fp2.close()
elif filename.split('.')[-1] == 'csv':
    fp2 = open(filename, 'r')
    rdr = csv.reader(fp2)
    gen_examples = []
    for ex in rdr:
        gen_examples.append((ex[0], ex[2]))
    fp2.close()
else:
    print >>sys.stderr, "Wrong file type for ", filename
    sys.exit(1)

example_type = sys.argv[4]

for ex in gen_examples:
    utterance = ex[0]
    target_json = ex[1]
    query = "insert into example_utterances " + \
                    " (language, type, utterance, target_json) values " + \
                    " ('%s', '%s', '%s', '%s')" % \
                    (MySQLdb.escape_string('en'), MySQLdb.escape_string(example_type), \
                     MySQLdb.escape_string(utterance), MySQLdb.escape_string(target_json))
    try:
        cursor.execute(query)
        #print query
    except MySQLdb.Error, e:
        try:
            print >>sys.stderr, "MySQL Error [%d]: %s" % (e.args[0], e.args[1])
        except IndexError:
            print >>sys.stderr, "MySQL Error: %s" % str(e)

conn.commit()
cursor.close()
