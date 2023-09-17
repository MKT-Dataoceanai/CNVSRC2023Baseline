# Data Preprocessing
We provide this pipeline to extract lip region videos from the provided video files and facelandmark files.
Please follow the steps below to download and preprocess the data.

1. Download the required dataset from the CNVSRC2023 website.

2. Modify the data paths in `run.sh` and execute `sh run.sh`.

## Process the downloaded `tar.gz` files



```bash
python crop_lip_video.py \
    --src $DOWNLOAD_DATA_PATH \
    --dst $TARGET_DATA_PATH \
    --csv $CODE_ROOT_PATH/data/$DATASET_NAME/$SPLIT.csv \
    --landmarks $DOWNLOAD_LANDMARK_PATH \
    --worker 8
```

The parameters for `crop_lip_video.py`` are as follows:

- `src`: The path to the downloaded dataset folder.
- `dst`: The target storage path of the processed video files, must not be the same as `src`, which will have the same structure as `src`.
- `csv`: The `.csv` file containing the path of videos to be processed, refer to [../data/multi-speaker/train.csv](../data/multi-speaker/train.csv).
- `landmarks`: The path to the downloaded and extracted facial landmark folder.
- `worker`: The number of threads used for multi-threaded processing.

You can modify the parameters in the bash file:

- `DOWNLOAD_DATA_PATH`: Path to the downloaded and extracted dataset.
- `TARGET_DATA_PATH`: The target path where the extracted video data will be stored. The directory structure will be the same as `$DOWNLOAD_DATA_PATH`.
- `CODE_ROOT_PATH`: The path to this code repository.
- `DATASET_NAME`: The name of the dataset to process, `cncvs` / `multi-speaker` / `single-speaker`.
- `SPLIT`: The dataset split to process, `train` / `valid`.
- `DOWNLOAD_LANDMARK_PATH`: The path to the downloaded and extracted facial landmark folder.

# Preprocessing Other Datasets in the Same Way

Please refer to the processing method provided by [Auto-AVSR](https://github.com/mpc001/auto_avsr/tree/main/preparation) for preprocessing other datasets.
