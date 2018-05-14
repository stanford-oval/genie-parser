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

        #hack
        list = [('''r"\'([^\'\\]|(\\(.|\n)))*?\'"'''), ('''r"\'([^\'\\]||(\\(.|\n)))*?\'"''')]
        for grep in list:
            if line.find(grep) != -1:
                line = line.replace(grep, 'STR')

        #if any(s in line for s in list):

        # if line.find('''r"\'([^\'\\]|(\\(.|\n)))*?\'"''') != -1:
        #     line = line.replace('''r"\'([^\'\\]|(\\(.|\n)))*?\'"''', 'STR')


        line = re.sub(r'\'\"\'', '\'\'', line)
        line = re.sub(r'\"\'\"', '\"\"', line)

        line = re.sub(r'\\\"(?!\s)', 'STR', line)
        line = re.sub(r'\\\'(?!\s)', 'STR', line)

        line = re.sub(r'n\'t', 'not', line)  # Empty string



        line = re.sub(r'b\"\"(?!\")|b\'\'(?!\')', ' STR ', line)  # Empty string
        line = re.sub(r'r\"\"(?!\")|r\'\'(?!\')', ' STR ', line)  # Empty string


        line = re.sub(r'r(\'{3}(.+?)\'{3})', r'\1', line)
        line = re.sub(r'r(\"{3}(.+?)\"{3})', r'\1', line)

        line = re.sub(r'r(\"(?!\"{2})(.+?)\")', r'\1', line)
        line = re.sub(r'r(\'(?!\'{2})(.+?)\')', r'\1', line)
        line = re.sub(r'b(\"(?!\"{2})(.+?)\")', r'\1', line)
        line = re.sub(r'b(\'(?!\'{2})(.+?)\')', r'\1', line)





        line = re.sub(r'\'{3}\s*\'{3}', ' STR ', line)  # Empty string
        line = re.sub(r'\"{3}\s*\"{3}', ' STR ', line)  # Empty string
        # line = re.sub(r'\'(?!\'{2})\s*\'(?!\'{2})', ' STR ', line)  # Empty string
        # line = re.sub(r'\"\s*\"', ' STR ', line)  # Empty string



        # line = re.sub(r'\'(?!\'{2})\s*\'', ' STR ', line)  # Empty string
        # line = re.sub(r'\"(?!\"{2})\s*\"', ' STR ', line)  # Empty string



        line = re.sub(r'\'\\\\\'', ' STR' + 'special', line)
        line = re.sub(r'\'\\\'', ' STR' + 'special', line)
        line = re.sub(r'\"_\(\'\"', ' STR' + 'special', line)
        line = re.sub(r'\"_\(\'\"', ' STR' + 'special', line)
        line = re.sub(r'\"_\(\'\"', ' STR' + 'special', line)
        line = re.sub(r'\"_\(\'\"', ' STR' + 'special', line)

        line = re.sub(r'0x|0d|0o', '', line)



        length1, length2, length3, length4 = (0, 0, 0, 0)

        length1 = len(re.findall(r'\'{3}(.+?)\'{3}', line))
        for i in range(length1):
            line = re.sub(r'\'{3}(.+?)\'{3}', ' STR' + str(i), line, count=1)

        length2 = len(re.findall(r'\"\"\"(.+?)\"\"\"', line)) + length1
        for i in range(length1, length2):
            line = re.sub(r'\"\"\"(?!STR)(.+?)\"\"\"', ' STR' + str(i), line, count=1)


        line = re.sub(r'\"\s*\"', ' STR ', line)  # Empty string



        length3 = len(re.findall(r'\"(.+?)\"', line)) + length2
        for i in range(length2, length3):
            line = re.sub(r'\"(?!STR)(.+?)\"', ' STR' + str(i), line, count=1)

        line = re.sub(r'\'\s*\'', ' STR ', line)  # Empty string

        length4 = len(re.findall(r'\'(.+?)\'', line)) + length3
        for i in range(length3, length4):
            line = re.sub(r'\'(?!STR)(.+?)\'', ' STR' + str(i), line, count=1)


        length1, length2, length3, length4 = (0, 0, 0, 0)

        length1 = len(re.findall(r'\'{3}(.+?)\'{3}', line))
        for i in range(length1):
            line = re.sub(r'\'{3}(.+?)\'{3}', ' STR' + str(i), line, count=1)

        length2 = len(re.findall(r'\"\"\"(.+?)\"\"\"', line)) + length1
        for i in range(length1, length2):
            line = re.sub(r'\"\"\"(?!STR)(.+?)\"\"\"', ' STR' + str(i), line, count=1)

        length3 = len(re.findall(r'\"(.+?)\"', line)) + length2
        for i in range(length2, length3):
            line = re.sub(r'\"(?!STR)(.+?)\"', ' STR' + str(i), line, count=1)

        length4 = len(re.findall(r'\'(.+?)\'', line)) + length3
        for i in range(length3, length4):
            line = re.sub(r'\'(?!STR)(.+?)\'', ' STR' + str(i), line, count=1)



        # cases like 'STR .... '
        length1, length2, length3, length4 = (0, 0, 0, 0)

        length1 = len(re.findall(r'\'{3}(.+?)\'{3}', line))
        for i in range(length1):
            line = re.sub(r'\'{3}(.+?)\'{3}', ' STR' + 'special', line, count=1)

        length2 = len(re.findall(r'\"\"\"(.+?)\"\"\"', line)) + length1
        for i in range(length1):
            line = re.sub(r'\"\"\"(.+?)\"\"\"', ' STR' + 'special', line, count=1)

        length3 = len(re.findall(r'\"(.+?)\"', line)) + length2
        for i in range(length1, length2):
            line = re.sub(r'\"(.+?)\"', ' STR' + 'special', line, count=1)

        length4 = len(re.findall(r'\'(.+?)\'', line)) + length3
        for i in range(length2, length3):
            line = re.sub(r'\'(.+?)\'', ' STR' + 'special', line, count=1)


        #hack
        line.replace("'STR'", "STR")
        line.replace("'STR '", "STR")
        line.replace("'STR STR8STR'", "STR, STR8")
        line.replace("'STR STR8STR '", "STR, STR8")

        line = re.sub(r'\'STR(.+?)\'', ' STR' + 'special', line)
        line.replace("_js_escapes = { ord ( STRspecial ) : STR0 , ord ( STRspecial", "_js_escapes = { ord ( STRspecial ) : STR0 , ord ( STRspecial )")




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