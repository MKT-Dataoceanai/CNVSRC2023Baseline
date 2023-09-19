# fill the blank and run
DOWNLOAD_DATA_PATH=
TARGET_DATA_PATH=
DATASET_NAME=
SPLIT=
# for example:
# DOWNLOAD_DATA_PATH='/data/CNVSRC/CNVSRC2023/'
# TARGET_DATA_PATH='/data/CNVSRC/CNVSRC2023_lips/'
# DATASET_NAME='cnvsrc-single'
# SPLIT='valid'
CODE_ROOT_PATH=$(dirname "$PWD")


if test -z "$DOWNLOAD_DATA_PATH"; then 
echo "DOWNLOAD_DATA_PATH is not set!"
exit 0
fi
if test -z "$TARGET_DATA_PATH"; then 
echo "TARGET_DATA_PATH is not set!"
exit 0
fi
if test -z "$DATASET_NAME"; then 
echo "DATASET_NAME is not set!"
exit 0
fi
if test -z "$SPLIT"; then 
echo "SPLIT is not set!"
exit 0
fi

python prepare_filescp.py \
    --src $DOWNLOAD_DATA_PATH \
    --dst $TARGET_DATA_PATH \
    --dataset $DATASET_NAME \
    --split $SPLIT

python detect_landmark_list.py --list $DATASET_NAME-$SPLIT.scp --rank 0 --shard 1
