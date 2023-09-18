# Data Preprocessing
We provide this pipeline to extract lip region videos from the provided video files.
Please follow the steps below to download and preprocess the data.

1. Download the required dataset from the CNVSRC2023 website.

2. Modify the data paths in `run.sh` and execute `sh run.sh`.

## Process the downloaded `tar.gz` files

By default, place the downloaded compressed file according to the following folder structure:
```
CNVSRC/
└── CNVSRC2023
    ├── cncvs/
    |   ├── news/
    |   |   ├── news_part01.tar.gz
    |   |   └── news_part02.tar.gz
    |   └── speech/
    |       ├── speech_part01.tar.gz
    |       ├── ...
    |       └── speech_part12.tar.gz
    ├── cnvsrc2023-ms-record-dev.tar.gz
    ├── cnvsrc2023-ms-vlog-dev.tar.gz
    └── cnvsrc2023-ss-dev.tar.gz
```

First, please execute the decompress command for all packages:

``` Shell
# cd CNVSRC/CNVSRC2023/
# cd cncvs/news/
# tar -xzvf news_part01.tar.gz
# tar -xzvf news_part02.tar.gz
# cd ../speech/
# tar -xzvf speech_part01.tar.gz
# tar -xzvf speech_part02.tar.gz
# tar -xzvf speech_part03.tar.gz
# tar -xzvf speech_part04.tar.gz
# tar -xzvf speech_part05.tar.gz
# tar -xzvf speech_part06.tar.gz
# tar -xzvf speech_part07.tar.gz
# tar -xzvf speech_part08.tar.gz
# tar -xzvf speech_part09.tar.gz
# tar -xzvf speech_part10.tar.gz
# tar -xzvf speech_part11.tar.gz
# tar -xzvf speech_part12.tar.gz
# cd ../../
# tar -xzvf cnvsrc2023-ms-record-dev.tar.gz --strip-components 7
# tar -xzvf cnvsrc2023-ms-vlog-dev.tar.gz --strip-components 7
# tar -xzvf cnvsrc2023-ss-dev.tar.gz --strip-components 7
```

After extracting the packages, you will get the following directory:

```
CNVSRC/
└── CNVSRC2023/
    ├── cncvs/
    |   ├── news/
    |   |   ├── n001/
    |   |   ├── ...
    |   |   └── n028/
    |   └── speech/
    |       ├── s00001/
    |       ├── ...
    |       └── s02529/
    ├── cnvsrc2023-ms-record-dev.tar.gz
    ├── cnvsrc2023-ms-vlog-dev.tar.gz
    ├── cnvsrc2023-ss-dev.tar.gz
    ├── ms-record/
    |   └── dev/
    |       ├── audio/
    |       └── video/
    ├── ms-vlog/
    |   └── dev/
    |       ├── audio/
    |       └── video/
    └── single-speaker/
        └── dev/
            ├── audio/
            └── video/
```

Then merge 'ms-record' and 'ms-vlog' with the following command:
```Shell
# mkdir -p CNVSRC/CNVSRC2023/multi-speaker/
# cp -r CNVSRC/CNVSRC2023/ms-record/dev/ CNVSRC/CNVSRC2023/multi-speaker/
# cp -r CNVSRC/CNVSRC2023/ms-vlog/dev/ CNVSRC/CNVSRC2023/multi-speaker/
```

Finally, you will get the following directory:
```
CNVSRC/
└── CNVSRC2023/
    ├── cncvs/
    ├── cnvsrc2023-ms-record-dev.tar.gz
    ├── cnvsrc2023-ms-vlog-dev.tar.gz
    ├── cnvsrc2023-ss-dev.tar.gz
    ├── ms-record/
    ├── ms-vlog/
    ├── multi-speaker/
    └── single-speaker/
```

Tips: Later the evaluation set will be available for download, for the downloaded evaluation set:
```
CNVSRC/
└── CNVSRC2023
    ├── cnvsrc2023-ms-record-eval.tar.gz
    ├── cnvsrc2023-ms-vlog-eval.tar.gz
    └── cnvsrc2023-ss-eval.tar.gz
```

All decompression and merge commands are the same as the dev set described above.

``` Shell
# cd CNVSRC/CNVSRC2023/
# tar -xzvf cnvsrc2023-ms-record-eval.tar.gz --strip-components 7
# tar -xzvf cnvsrc2023-ms-vlog-eval.tar.gz --strip-components 7
# tar -xzvf cnvsrc2023-ss-eval.tar.gz --strip-components 7
```


## How to run `run.sh`

The point of 'run.sh' is to call `prepare_filescp.py` and `detect_landmark_list.py` reasonably:

```Shell
python prepare_filescp.py \
    --src $DOWNLOAD_DATA_PATH \
    --dst $TARGET_DATA_PATH \
    --dataset $DATASET_NAME \
    --split $SPLIT

python detect_landmark_list.py --list $DATASET_NAME-$SPLIT.scp --rank 0 --shard 1
```

The `prepare_filescp.py` will generate the `.scp` list file required for the second step based on the downloaded dataset path and the target path.

`detect_landmark_list.py` will detect and extract videos of lips based on `.scp` list files.

You can run it by modifying the parameters in the bash file:

- `DOWNLOAD_DATA_PATH`: Path to the downloaded and extracted dataset, eg. `CNVSRC/CNVSRC2023/`.
- `TARGET_DATA_PATH`: The target path where the extracted video data will be stored. The directory structure will be the same as `$DOWNLOAD_DATA_PATH`, we recommand using `CNVSRC/CNVSRC2023_lips/`.
- `DATASET_NAME`: The name of the dataset to process, `cncvs` / `multi-speaker` / `single-speaker`.
- `SPLIT`: The dataset split to process, `train` / `valid`, this parameter has no effect when processing the `CNCVS` dataset. 
- `CODE_ROOT_PATH`: The path to this code repository.

Note: For the `cncvs` dataset, set the `SPLIT` to `train` to preprocess all data. There is no need to preprocess `valid` for `cncvs`.

# Preprocessing Other Datasets in the Same Way

Please refer to the processing method provided by [Auto-AVSR](https://github.com/mpc001/auto_avsr/tree/main/preparation) for preprocessing other datasets.
