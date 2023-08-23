
# 数据预处理

我们为根据已提供的视频文件和人脸特征点文件提取口唇部位视频文件提供了处理代码。请依照下面的流程下载并对数据进行预处理。

1. 从挑战赛官网下载所需数据集。

2. 从挑战赛官网下载数据集对应的人脸特征点文件。

3. 修改`run.sh`中的数据路径，并执行`sh run.sh`

```Shell
python crop_lip_video.py \
    --src $DOWNLOAD_DATA_PATH \
    --dst $TARGET_DATA_PATH \
    --csv $CODE_ROOT_PATH/data/$DATASET_NAME/$SPLIT.csv \
    --landmarks $DOWNLOAD_LANDMARK_PATH \
    --worker 8
```
`crop_lip_video.py`所需的参数含义为：
- `src`: 下载并解压后的数据集文件夹路径。
- `dst`: 目标存储路径，**不得与`src`相同**，处理后其将会与src具有相同的目录结构。
- `csv`: 包含要处理的视频的路径和其他信息，参照[../data/multi-speaker/train.csv](../data/multi-speaker/train.csv)及[README](../README.md)中的介绍。
- `landmarks`: 下载并解压后的人脸特征点文件夹路径。
- `worker`: 多线程处理所用的线程数量。

您可以通过修改bash文件中的参数来运行：

- `DOWNLOAD_DATA_PATH`: 已下载并解压的的数据集路径。
- `TARGET_DATA_PATH`: 目标路径，已提取出的视频数据将会储存在目标路径下，其目录结构与`$DOWNLOAD_DATA_PATH`保持一致。
- `CODE_ROOT_PATH`: 本代码库的路径。
- `DATASET_NAME`: 想要处理的数据集的名字，`cncvs` / `multi-speaker` / `single-speaker`
- `SPLIT`: 想要处理的数据集划分，`train` / `valid`
- `DOWNLOAD_LANDMARK_PATH`: 下载并解压后的人脸特征点文件夹路径。

# 使用相同方式处理其他数据集

请参照[Auto-AVSR](https://github.com/mpc001/auto_avsr/tree/main/preparation)提供的处理方式进行预处理。
