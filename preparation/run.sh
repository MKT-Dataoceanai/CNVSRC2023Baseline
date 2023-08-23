# fill the blank and run
# DOWNLOAD_DATA_PATH=
# DOWNLOAD_LANDMARK_PATH=
# TARGET_DATA_PATH=
# DATASET_NAME=
# SPLIT=
# for example:
# DOWNLOAD_DATA_PATH='/data/CNVSRC/'
# DOWNLOAD_LANDMARK_PATH='/data/CNVSRC_landmarks/'
# TARGET_DATA_PATH='/data/CNVSRC_lips/'
# DATASET_NAME='multi-speaker'
# SPLIT='valid'
DOWNLOAD_DATA_PATH='/work101/cchen/data/CNVSRC/'
DOWNLOAD_LANDMARK_PATH='/work101/cchen/data/CNVSRC_landmarks/'
TARGET_DATA_PATH='/work101/cchen/data/CNVSRC_lips/'
DATASET_NAME='multi-speaker'
SPLIT='valid'
CODE_ROOT_PATH=$(dirname "$PWD")


if test -z "$DOWNLOAD_DATA_PATH"; then 
echo "DOWNLOAD_DATA_PATH is not set!"
exit 0
fi
if test -z "$DOWNLOAD_LANDMARK_PATH"; then 
echo "DOWNLOAD_LANDMARK_PATH is not set!"
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

python crop_lip_video.py \
    --src $DOWNLOAD_DATA_PATH \
    --dst $TARGET_DATA_PATH \
    --csv $CODE_ROOT_PATH/data/$DATASET_NAME/$SPLIT.csv \
    --landmarks $DOWNLOAD_LANDMARK_PATH \
    --worker 8