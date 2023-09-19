<h1 align="center">CNVSRC2023 基线系统</h1>

## 简介

本仓库是CNVSRC2023挑战赛的基线系统代码。

本仓库的代码源自LRS3数据集上的SOTA方法[mpc001/auto_avsr](https://github.com/mpc001/auto_avsr)。我们添加了一些配置文件以使用此代码在CN-CVS和本次挑战赛各任务中提供的数据集上进行训练。此外，我们删除了部分运行此baseline时暂不需要的代码并修改了部分功能的实现方式。

## 准备工作

1. 克隆本仓库并进入代码路径:

```Shell
git clone git@github.com:MKT-Dataoceanai/CNVSRC2023Baseline.git
cd CNVSRC2023Baseline
git submodule init
git submodule update
cd tools/face_detection
git lfs pull
cd ../../
```

如果您没有安装git lfs，请到[这里](https://github.com/sectum1919/face_detection/tree/ec0d6be271871f4ec551d82c2b6c55779d9d60db/ibug/face_detection/retina_face/weights)下载模型文件并放入`CNVSRC2023Baseline/tools/face_detection/ibug/face_detection/retina_face/weights/`

|       Model name      |             md5sum             |
|-----------------------|--------------------------------|
|Resnet50_Final.pth     |bce939bc22d8cec91229716dd932e56e|
|mobilenet0.25_Final.pth|d308262876f997c63f79c7805b2cdab0|

2. 使用Conda创建虚拟环境并安装依赖库:

```Shell
# create environment
conda create -y -n cnvsrc python==3.10.11
conda activate cnvsrc
# install pytorch torchaudio torchvision
conda install pytorch-lightning==1.9.3 pytorch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2 torchmetrics==0.11.2 pytorch-cuda==11.8 cudatoolkit==11.8 -c pytorch -c nvidia -y
# install fairseq
cd tools/fairseq/
pip install --editable .
# install face_alignment
cd ../face_alignment/
pip install --editable .
# install face_detection
cd ../face_detection/
pip install --editable .
# install other dependence
cd ../../
pip install -r reqs.txt
```

3. 下载并处理挑战赛提供的数据集。请参考[preparation](./preparation)文件夹中提供的详细指导。

## Logging

我们使用 tensorboard 作为日志工具。运行时相应的文件将会被写入`outputs/$DATE/$TIME/tblog/`路径中。

## 训练

请修改`main.py`中指定的`yaml`配置文件来选择目标的训练配置。

### 配置文件

文件夹[conf](conf/)中列出了本baseline所需的配置参数。

在运行任何训练和测试之前，请务必将对应的`yaml`文件中的`code_root_dir`和`data_root_dir`分别修改为本仓库路径和数据所在路径。

`data.dataset`指定了所用数据集的路径和列表文件路径。

以`cncvs`为例：

1. 当数据预处理完成后，请将`data_root_dir`设置为`cncvs/`的上级目录，并将[data/cncvs/*.csv](data/cncvs/test.csv)复制到`${data_root_dir}/cncvs/*.csv`。

2. 此时，`${data_root_dir}`的文件夹结构如：
   
   ```
   ${data_root_dir}
   └── cncvs
       ├── test.csv
       ├── train.csv
       ├── valid.csv
       ├── news
       |   ├── n001
       |   |   └── video
       |   |       ├── ...
       |   |       └── n001_00001_001.mp4
       |   ├── n002
       |   ├── ...
       |   └── n028
       └── speech
           ├── s00001
           ├── s00002
           ├── ...
           └── s02529
   ```
3. 此时`*.csv`的内容示例为：
   ```
   cncvs,news/n001/video/n001_00001_001.mp4,x,x x x x x x x x x x x
   ```
   其中每一行的内容为
   ```
   ${dataset_name},${video_filename},${frame_count},${token_idx_split_by_blank}
   ```

### 使用预训练模型进行微调

配置文件[train_multi-speaker.yaml](conf/train_multi-speaker.yaml)给出了一个使用CN-CVS预训练模型进行微调的配置示例。

配置文件中`ckpt_path`指定了预训练模型的路径。`remove_ctc`表示是否使用预训练模型的分类层。

```Shell
python main.py
```

### 通过课程学习从零开始训练

从零开始训练模型通常需要使用课程学习的方式，即先使用不超过4s的数据对模型进行初步训练，再用得到的模型初始化之后用全部数据进行训练。

**[Stage 1]** 使用不超过4s的数据训练模型。

可以通过配置文件中的`data.max_length=100`来确保模型在训练时仅使用不超过100帧（4s）的数据进行训练。

**[Stage 2]** 使用Stage 1 得到的模型初始化并使用全部数据进行训练。

## 测试

使用`main.py`和`predict.py`均可进行测试，通过指定配置文件或修改配置文件中的`data.dataset.test_file`来选择测试文件列表。

推荐使用`predict.py`实时观察输出的文字信息。

## Model zoo

下面的表格列出了各个模型在各自任务上的CER。

可以从 [huggingface](https://huggingface.co/DataOceanAI/CNVSRC2023Baseline) 或 [modelscope](https://www.modelscope.cn/speechoceanadmin/CNVSRC2023Baseline) 下载模型文件。

|          Task         |       Training Data           | CER on Dev | CER on Eval | File Name                                |
|:---------------------:|:-----------------------------:|:----------:|:-----------:|:-----------------------------------------|
|     Pre-train         | CN-CVS (<4s)                  |     /      |      /      | model_avg_14_23_cncvs_4s.pth             |
|     Pre-train         | CN-CVS (full)                 |     /      |      /      | model_avg_last10_cncvs_4s_30s.pth        |
|Single-speaker VSR (T1)| CN-CVS + CNVSRC-Single.Dev    |   48.57%   |    48.60%   | model_avg_last5_cncvs_cnvsrc-single.pth  |
|Multi-speaker VSR (T2) | CN-CVS + CNVSRC-Multi.Dev     |   58.77%   |    58.37%   | model_avg_last5_cncvs_cnvsrc-multi.pth   |

## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./LICENSE) for non-commercial purposes.

## Contact

```
[Chen Chen](chenchen[at]cslt.org)
```
