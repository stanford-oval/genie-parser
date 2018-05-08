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
        #line.replace('"', "'")

        if os.path.basename(inputfile) == 'all.anno':

            length1 = len(re.findall(r'\"(.*?)\"', line))
            for i in range(length1):
                line = re.sub(r'\"(?!STR)(.*?)\"', 'STR' + str(i), line, count=1)

            length2 = len(re.findall(r'\'(.*?)\'', line)) + length1
            for i in range(length1,length2):
                line = re.sub(r'\'(?!STR)(.*?)\'', 'STR' + str(i), line, count=1)


            tokens = word_tokenize(line)
            print(' '.join(tokens), end='\n', file=fout)


        elif os.path.basename(inputfile) == 'all.code':

            length1 = len(re.findall(r'\"(.*?)\"', line))
            for i in range(length1):
                line = re.sub(r'\"(?!STR)(.*?)\"', 'STR' + str(i), line, count=1)


            length2 = len(re.findall(r'\'(.*?)\'', line)) + length1
            for i in range(length1,length2):
                line = re.sub(r'\'(?!STR)(.*?)\'', 'STR' + str(i), line, count=1)


            tokens = line.strip().split(' ')
            print(' '.join(tokens), end='\n', file=fout)






