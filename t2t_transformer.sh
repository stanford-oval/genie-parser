PROBLEM=parse_almond
MODEL=transformer
HPARAMS=transformer_base_single_gpu

PROBLEM_DIR=$HOME/almond-nnparser/
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

## Generate data
#t2t-datagen \
#  --t2t_user_dir=$PROBLEM_DIR \
#  --data_dir=$DATA_DIR \
#  --tmp_dir=$TMP_DIR \
#  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer --t2t_usr_dir=$PROBLEM_DIR --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR --model=$MODEL --hparams_set=$HPARAMS --output_dir=$TRAIN_DIR --problem=$PROBLEM --hparams='batch_size=512'
