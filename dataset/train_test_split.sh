#!/bin/bash


## run example: sudo ./train_test_split.sh ./ 100 500##

DATA=$1
ratio=$2
keep=$3


python3 $DATA/tokenize_data.py $DATA/ $DATA/all.anno
python3 $DATA/tokenize_data.py $DATA/ $DATA/all.code


sed 's/^ *//g' < $DATA/tokens_all.code.txt> $DATA/code++.txt
#sed 's/^ *//g' < $DATA/all.anno > $DATA/anno++.txt

nl -w 5  $DATA/tokens_all.anno.txt > $DATA/anno_num.txt
#nl -w 5  -s ' ' $DATA/code++.txt > $DATA/code_num.txt

sed 's/^ *//g' < $DATA/anno_num.txt > $DATA/anno_num++.txt
#sed 's/^ *//g' < $DATA/code_num.txt > $DATA/code_num++.txt



paste $DATA/anno_num++.txt $DATA/code++.txt > $DATA/data.txt

gshuf -o $DATA/data_shuffled.txt < $DATA/data.txt
lines=`cat $DATA/data_shuffled.txt | wc -l`
trainlines=$(( lines / ratio ))
csplit -s $DATA/data_shuffled.txt $trainlines

mv xx00  test.txt
mv xx01  train.txt

head -n $keep $DATA/train.txt > $DATA/train_keep.txt
head -n $keep $DATA/test.txt > $DATA/test_keep.txt

cut -f3 $DATA/train_keep.txt > $DATA/code_keep.txt
cut -f3 $DATA/test_keep.txt >> $DATA/code_keep.txt





