<h1 align="center">CNVSRC 2023 Baseline</h1>

## Introduction

This repository is the baseline code for CNVSRC2023 (Chinese Continuous Visual Speech Recognition Challenge 2023).

The code in this repository is based on the SOTA method [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr) on the LRS3 dataset. We have added some configuration files to run this code on CN-CVS and the datasets provided in this challenge. Additionally, we have removed some code that is not needed for running this baseline and modified the implementation of some functionalities.

## Preparation

1. Clone the repository and navigate to it:

```Shell
git clone git@github.com:MKT-Dataoceanai/CNVSRC2023Baseline.git
cd CNVSRC2023Baseline
git submodule init
git submodule update
cd tools/face_detection
git lfs pull
cd ../../
```

If you don't have git-lfs installed, Please go to [here](https://github.com/sectum1919/face_detection/tree/ec0d6be271871f4ec551d82c2b6c55779d9d60db/ibug/face_detection/retina_face/weights) to download the model file and put it into `CNVSRC2023Baseline/tools/face_detection/ibug/face_detection/retina_face/weights/`

|       Model name      |             md5sum             |
|-----------------------|--------------------------------|
|Resnet50_Final.pth     |bce939bc22d8cec91229716dd932e56e|
|mobilenet0.25_Final.pth|d308262876f997c63f79c7805b2cdab0|

2. Set up the environment:

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

3. Download and preprocess the dataset. See the instructions in the [preparation](./preparation) folder.

4. Download the [models](#Model-zoo) into path [pretrained_models/](pretrained_models/)

## Logging

We use tensorboard as the logger.

Tensorboard logging files will be writen to `outputs/$DATE/$TIME/tblog/`

## Training

Please modify the specified `yaml` configuration file in `main.py` to select the training configuration.

### Configuration File

The [conf](conf/) folder lists the configuration files required for this baseline.

Before running any training or testing, please make sure to modify the `code_root_dir` and `data_root_dir` in the corresponding `yaml` file to **the path of this repository** and **the path where the dataset is located**, respectively.

`data.dataset` specifies the path of the dataset and the path of the `.csv` files.

Taking `cncvs` as an example:

1. After data preprocessing is complete, please set `${data_root_dir}` to the parent directory of `cncvs/` and copy [data/cncvs/*.csv](data/cncvs/test.csv) to `${data_root_dir}/cncvs/*.csv`.

2. At this point, the folder structure of `${data_root_dir}` is as follows:

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

3. The contents of `*.csv` at this point are as follows:
   ```
   cncvs,news/n001/video/n001_00001_001.mp4,x,x x x x x x x x x x x
   ```
   Each line contains the following content:
   ```
   ${dataset_name},${video_filename},${frame_count},${token_idx_split_by_blank}
   ```


### Training from a pre-trained model

The configuration file [train_cnvsrc-multi.yaml](conf/train_cnvsrc-multi.yaml) provides an example configuration for fine-tuning a model pretrained on CN-CVS.

The `ckpt_path` in the configuration file specifies the path of the pretrained model. The `remove_ctc` option indicates whether to use the classification layer of the pretrained model.

```Shell
python main.py
```

### Training from scratch through curriculum learning

**[Stage 1]** Train the model using the subset that includes only short utterances lasting no more than 4 seconds (100 frames). We set `optimizer.lr` to 0.0002 at the first stage.

By setting `data.max_length=100` in config files to ensure training use only short utterances. 

See [train_cncvs_4s.yaml](conf/train_cncvs_4s.yaml) for more information.

**[Stage 2]** Use the checkpoint from stage 1 to initialise the model and train the model with the full dataset.

See [train_cncvs_4s_30s.yaml](conf/train_cncvs_4s_30s.yaml) for more information.

## Testing

You can use either `main.py` or `predict.py` to run testing as soon as you choose the proper config filename in the python file.

We prefer using `predict.py` to observe the real-time output text.

## Model zoo

The table below contains CER on their own task.

Download model files from [huggingface](https://huggingface.co/DataOceanAI/CNVSRC2023Baseline) or [modelscope](https://www.modelscope.cn/speechoceanadmin/CNVSRC2023Baseline)。

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
