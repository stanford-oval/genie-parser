import sys
import os
import re
from nltk.tokenize import word_tokenize


'''
split standard contractions, e.g. don't -> do n't and they'll -> they 'll

treat most punctuation characters as separate tokens

split off commas and single quotes, when followed by whitespace

separate periods that appear at the end of line
'''

if len(sys.argv) != 3:
    print(
        "** Usage: python3 " + "/path/to/tokenize_data.py" + "workdir path" + "dataset file **")
    sys.exit(1)
else:
    workdir, inputfile = sys.argv[1:3]
with open(inputfile, 'r') as fin, open(os.path.join(workdir, 'tokens_' + os.path.basename(inputfile) + '.txt'), 'w') as fout:

    for line in fin:



        # exceptions = [('''if bit starts with '_("' ot bit starts with "_('",'''), ('''if bit . startswith ( '_("' ) or bit . startswith ( "_('" ) :''')]
        #
        # if line in exceptions:
        #
        #     line = re.sub(r'\'_\(\"\'', 'STR' + str(0), line, count=1)
        #     line = re.sub(r'\"_\(\'\"', 'STR' + str(1), line, count=1)
        #
        #


        # else:

        #line = re.sub(r'\"|\'\'\'', '\'', line)

        #line = re.sub(r'rSTR|bSTR|\\STR|b\'\'|r\'\'', 'STR', line)
        line = re.sub(r'\'t', 'ot', line)  # Empty string

        line = re.sub(r'b\"\"|b\'\'', ' STR ', line)  # Empty string
        line = re.sub(r'r\"\"|r\'\'', ' STR ', line)  # Empty string

        line = re.sub(r'r(\"(?!\"{2})(.+?)\")', r'\1', line)
        line = re.sub(r'r(\'(?!\'{2})(.+?)\')', r'\1', line)
        line = re.sub(r'b(\"(?!\"{2})(.+?)\")', r'\1', line)
        line = re.sub(r'b(\'(?!\'{2})(.+?)\')', r'\1', line)

        line = re.sub(r'\'(?!\'{2})\s*\'', ' STR ', line)  # Empty string
        line = re.sub(r'\"(?!\"{2})\s*\"', ' STR ', line)  # Empty string
        line = re.sub(r'\'_\(\"\'', ' STR' + 'special', line)
        line = re.sub(r'\"_\(\'\"', ' STR' + 'special', line)
        line = re.sub(r'0x|0d', '', line)


        length1 = len(re.findall(r'\"\"\"(.+?)\"\"\"', line))
        for i in range(length1):
            line = re.sub(r'\"\"\"(?!STR)(.+?)\"\"\"', ' STR' + str(i), line, count=1)

        length2 = len(re.findall(r'\"(.+?)\"', line)) + length1
        for i in range(length1, length2):
            line = re.sub(r'\"(?!STR)(.+?)\"', ' STR' + str(i), line, count=1)

        length3 = len(re.findall(r'\'(.+?)\'', line)) + length2
        for i in range(length2, length3):
            line = re.sub(r'\'(?!STR)(.+?)\'', ' STR' + str(i), line, count=1)


        length1, length2, length3 = (0, 0, 0)

        length1 = len(re.findall(r'\"\"\"(.+?)\"\"\"', line))
        for i in range(length1):
            line = re.sub(r'\"\"\"(?!STR)(.+?)\"\"\"', ' STR' + str(i), line, count=1)


        length2 = len(re.findall(r'\"(.+?)\"', line)) + length1
        for i in range(length1, length2):
            line = re.sub(r'\"(?!STR)(.+?)\"', ' STR' + str(i), line, count=1)

        length3 = len(re.findall(r'\'(.+?)\'', line)) + length2
        for i in range(length2, length3):
            line = re.sub(r'\'(?!STR)(.+?)\'', ' STR' + str(i), line, count=1)



        # cases like 'STR .... '
        length1, length2, length3 = (0, 0, 0)

        length1 = len(re.findall(r'\"\"\"(.+?)\"\"\"', line))
        for i in range(length1):
            line = re.sub(r'\"\"\"(.+?)\"\"\"', ' STR' + 'special', line, count=1)


        length2 = len(re.findall(r'\"(.+?)\"', line)) + length1
        for i in range(length1, length2):
            line = re.sub(r'\"(.+?)\"', ' STR' + 'special', line, count=1)

        length3 = len(re.findall(r'\'(.+?)\'', line)) + length2
        for i in range(length2, length3):
            line = re.sub(r'\'(.+?)\'', ' STR' + 'special', line, count=1)




        if os.path.basename(inputfile) == 'all.anno':

            tokens = word_tokenize(line)
            print(' '.join(tokens), end='\n', file=fout)


        elif os.path.basename(inputfile) == 'all.code':


            # line = re.sub(r'rSTR|bSTR|\\STR|b\'\'|r\'\'', 'STR', line)
            # line = re.sub(r'\'\s*\'', ' STR ', line)  # Empty string
            # line = re.sub(r'\"\s*\"', ' STR ', line)  # Empty string


            tokens = line.strip().split(' ')
            tokens = [token for token in tokens if token != '']
            print(' '.join(tokens), end='\n', file=fout)