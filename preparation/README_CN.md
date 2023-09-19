
# 数据预处理

我们为根据已提供的视频文件提取口唇部位视频文件提供了处理代码。请依照下面的流程下载并对数据进行预处理。

1. 从挑战赛官网下载所需数据集。

2. 修改`run.sh`中的数据路径，并执行`sh run.sh`

## 如何处理下载得到的压缩文件

我们默认您将下载得到的压缩文件按照以下文件夹结构放置：
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
    ├── cnvsrc-multi-dev.tar.gz
    └── cnvsrc-single-dev.tar.gz
```

请首先对所有压缩包执行解压缩命令
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
# tar -xzvf cnvsrc-multi-dev.tar.gz
# tar -xzvf cnvsrc-single-dev.tar.gz

```
解压全部压缩包后，您将得到以下目录：

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
    ├── cnvsrc-multi/
    │   ├── cnvsrc-multi-dev-infos.json
    │   ├── README.TXT
    |   └── dev/
    |       ├── audio/
    |       └── video/
    ├── cnvsrc-multi-dev.tar.gz
    ├── cnvsrc-single/
    |   ├── cnvsrc-single-dev-infos.json
    |   ├── README.TXT
    |   └── dev/
    |       ├── audio/
    |       └── video/
    └── cnvsrc-single-dev.tar.gz
```

Tips: 在竞赛的后期会开放下载评测集，对于下载得到评测集：
```
CNVSRC/
└── CNVSRC2023
    ├── cnvsrc-multi-eval.tar.gz
    └── cnvsrc-single-eval.tar.gz
```
所有的解压缩操作均与上述的开发集相同，即：

``` Shell
# cd CNVSRC/CNVSRC2023/
# tar -xzvf cnvsrc-multi-eval.tar.gz
# tar -xzvf cnvsrc-single-eval.tar.gz
```

此时将得到以下目录结构：

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
    ├── cnvsrc-multi/
    │   ├── cnvsrc-multi-dev-infos.json
    |   ├── cnvsrc-multi-eval-infos.json
    │   ├── README.TXT
    |   ├── dev/
    |   |   ├── audio/
    |   |   └── video/
    |   └── eval/
    |       └── video/
    ├── cnvsrc-multi-dev.tar.gz
    ├── cnvsrc-multi-eval.tar.gz
    ├── cnvsrc-single/
    |   ├── cnvsrc-single-dev-infos.json
    |   ├── cnvsrc-single-eval-infos.json
    |   ├── README.TXT
    |   ├── dev/
    |   |   ├── audio/
    |   |   └── video/
    |   └── eval/
    |       └── video/
    ├── cnvsrc-single-dev.tar.gz
    └── cnvsrc-single-eval.tar.gz
```

## 如何运行`run.sh`

`run.sh`的重点在于合理地调用`prepare_filescp.py`和`detect_landmark_list.py`:

```Shell
python prepare_filescp.py \
    --src $DOWNLOAD_DATA_PATH \
    --dst $TARGET_DATA_PATH \
    --dataset $DATASET_NAME \
    --split $SPLIT

python detect_landmark_list.py --list $DATASET_NAME-$SPLIT.scp --rank 0 --shard 1
```

其中`prepare_filescp.py`会根据下载好的数据集路径，以及设置好的目标路径，生成第二步所需的scp列表文件。

而`detect_landmark_list.py`则将根据scp列表文件检测并抽取口唇部位的视频。

您可以通过修改bash文件中的参数来运行：

- `DOWNLOAD_DATA_PATH`: 已下载并解压的的数据集路径，按照上述的解压方式时应为`CNVSRC/CNVSRC2023/`
- `TARGET_DATA_PATH`: 目标路径，已提取出的视频数据将会储存在目标路径下，其目录结构与`$DOWNLOAD_DATA_PATH`保持一致，推荐使用`CNVSRC/CNVSRC2023_lips/`。
- `DATASET_NAME`: 想要处理的数据集的名字，`cncvs` / `multi-speaker` / `single-speaker`
- `SPLIT`: 想要处理的数据集划分，`train` / `valid`，这个参数对于`cncvs`数据集不起作用。
- `CODE_ROOT_PATH`: 本代码库的路径，如果您在本目录下直接执行此bash文件则不需要改动。

Note: 对于`cncvs`数据集，`SPLIT`参数设置为`train`即可对所有数据均进行预处理，不需要再对`valid`进行预处理。

# 使用相同方式处理其他数据集

请参照[Auto-AVSR](https://github.com/mpc001/auto_avsr/tree/main/preparation)提供的处理方式进行预处理。
