#!/bin/bash


## run example: sudo ./train_test_split.sh ./ ##

DATA=$1


python3 $DATA/../tokenize_data.py $DATA/ $DATA/../all.anno
python3 $DATA/../tokenize_data.py $DATA/ $DATA/../all.code


sed 's/^ *//g' < $DATA/tokens_all.code.txt> $DATA/code++.txt
#sed 's/^ *//g' < $DATA/all.anno > $DATA/anno++.txt

nl -w 5  $DATA/tokens_all.anno.txt > $DATA/anno_num.txt
#nl -w 5  -s ' ' $DATA/code++.txt > $DATA/code_num.txt

sed 's/^ *//g' < $DATA/anno_num.txt > $DATA/anno_num++.txt
#sed 's/^ *//g' < $DATA/code_num.txt > $DATA/code_num++.txt



paste $DATA/anno_num++.txt $DATA/code++.txt > $DATA/data.txt



sed -n 1,15998p $DATA/data.txt > $DATA/train.txt

sed -n 15999,16998p $DATA/data.txt > $DATA/valid.txt

sed -n 16999,18805p $DATA/data.txt > $DATA/test.txt

# sed -i 's/great15914/great/g' $DATA/train.txt
# sed -i 's/y13277/y/g' $DATA/train.txt

# gshuf -o $DATA/data_shuffled.txt < $DATA/data.txt
# lines=`cat $DATA/data_shuffled.txt | wc -l`
# trainlines=$(( lines / ratio ))
# csplit -s $DATA/data_shuffled.txt $trainlines

# mv xx00  test.txt
# mv xx01  train.txt

# head -n $keep $DATA/train.txt > $DATA/train_keep.txt
# head -n $keep $DATA/test.txt > $DATA/test_keep.txt

# cut -f3 $DATA/train_keep.txt > $DATA/code_keep.txt
# cut -f3 $DATA/test_keep.txt >> $DATA/code_keep.txt





