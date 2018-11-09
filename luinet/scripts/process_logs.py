#!/usr/bin/python3
#
# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
#
# Author: Mehrad Moradshahi <mehrad@cs.stanford.edu>
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
Created on Nov 5, 2018

@author: mehrad
'''

import sys
import os
import re
import argparse
import csv

parser = argparse.ArgumentParser()

parser.add_argument('--workdir', type=str)
parser.add_argument('--output_file', type=str)

args = parser.parse_args()

if os.path.exists(args.output_file):
    os.remove(args.output_file)

if __name__ == '__main__':

    csvRow = ['Device', 'Accuracy', 'Accuracy w/o params', 'BLEU', 'Function', 'Grammar', 'Prim/Compound', 'Token F1']
    with open(args.output_file, "w") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(csvRow)

        for file in os.listdir(args.workdir):
            print(file)
            if file.startswith('.'):
                continue
            max_accuracy = 0.0
            max_line = ''
            with open(os.path.join(os.path.join(args.workdir, file), 'log'), 'r') as log:
                for line in log:
                    if line.startswith('INFO:tensorflow:Saving dict for global step'):
                        acc = float(re.findall('/accuracy = (\d+\.\d*)', line)[0])
                        if acc >= max_accuracy:
                            max_accuracy = acc
                            max_line = line
            # extract metrics
            accuracy = float(re.findall('/accuracy = (\d+\.\d*)', max_line)[0])
            accuracy_without_parameters = float(re.findall('/accuracy_without_parameters = (\d+\.\d*)', max_line)[0])
            bleu_score = float(re.findall('/bleu_score = (\d+\.\d*)', max_line)[0])
            function_accuracy = float(re.findall('/function_accuracy = (\d+\.\d*)', max_line)[0])
            grammar_accuracy = float(re.findall('/grammar_accuracy = (\d+\.\d*)', max_line)[0])
            num_function_accuracy = float(re.findall('/num_function_accuracy = (\d+\.\d*)', max_line)[0])
            token_f1_accuracy = float(re.findall('/token_f1_accuracy = (\d+\.\d*)', max_line)[0])

            wr.writerow([file, "{0:.2%}".format(accuracy), "{0:.2%}".format(accuracy_without_parameters), "{0:.2%}".format(bleu_score), "{0:.2%}".format(function_accuracy), "{0:.2%}".format(grammar_accuracy), "{0:.2%}".format(num_function_accuracy), "{0:.2%}".format(token_f1_accuracy)])

    # sort csv file by device name
    with open(args.output_file) as csvfile:
        reader = csv.DictReader(csvfile)
        sortedlist = sorted(reader, key=lambda row: row['Device'], reverse=False)

    with open(os.path.join(os.path.dirname(args.output_file), 'sorted_{}.csv'.format(os.path.basename(args.output_file).split('_')[-1][:-len('.csv')])), 'w') as f:
        fieldnames = ['Device', 'Accuracy', 'Accuracy w/o params', 'BLEU', 'Function', 'Grammar', 'Prim/Compound', 'Token F1']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sortedlist:
            writer.writerow(row)
            writer.writerow({k: '' for k in row.keys()})
            writer.writerow({k: '' for k in row.keys()})


