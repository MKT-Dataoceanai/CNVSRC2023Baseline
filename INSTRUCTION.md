## Training on other datasets

### Step 1. Training a sentencepiece model

- We have included the SentencePiece model we used for English corpus and the corresponding paths below, which are used in `TextTransform` class included in [preparation/transforms.py](preparation/transforms.py) and [datamodule/transforms.py](datamodule/transforms.py).

|              File Path                  |            Hash Value             |
| --------------------------------------- | --------------------------------- |
| `spm/unigram/unigram5000_units.txt`     | e652da86609085b8f77e5cffcd1943bd  |
| `spm/unigram/unigram5000.model`         | f2f6e8407b86538cf0c635a534eda799  |

- If the language spoken is not English or the content is substantially different from the LRS3 content, you will not be able to use our provided SentencePiece model derived from LRS3. In this case, you will need to train a new SentencePiece model. To do this, please start by customizing the input file [spm/input.txt](./spm/input.txt) with your training corpus.Once completed, run the script [spm/input.txt](./spm/input.txt). If you decide to retrain the SentencePiece model, please ensure to update the corresponding paths for `SP_MODEL_PATH` and `DICT_PATH` in [preparation/transforms.py](preparation/transforms.py) and [datamodule/transforms.py](datamodule/transforms.py).

### Step 2. Building a pre-processed dataset

- We provide a directory structure for a custom dataset `cstm` as below. The `preprocess_datasets/cstm` folder stores pre-processed audio-video-text pairs, while `preprocess_datasets/labels` stores a label list file. Here are the steps for creating both folders:

    ```
    preprocess_datasets/
    │
    ├── cstm/
    │ ├── cstm_text_seg24s/
    │ │ ├── file_1.txt
    │ │ └── ...
    │ │
    │ └── cstm_video_seg24s/
    │ ├── file_1.mp4
    │ ├── file_1.wav
    │ └── ...
    │
    ├── labels/
    │ ├── cstm_transcript_lengths_seg24s.csv
    ```

- Code snippts below to save pre-processed audio-visual pairings and their corresponding text files:

    ```Python
    from preparation.data.data_module import AVSRDataLoader
    from preparation.utils import save_vid_aud_txt

    # Initialize video and audio data loaders
    video_loader = AVSRDataLoader(modality="video", detector="retinaface", convert_gray=False)
    audio_loader = AVSRDataLoader(modality="audio")

    # Specify the file path to the data
    data_path = 'data_filename'

    # Load video and audio data from the same data file
    video_data = video_loader.load_data(data_path)
    audio_data = audio_loader.load_data(data_path)

    # Load text
    text = ...

    # Define output paths for the processed video, audio, and text data
    output_video_path = 'cstm/cstm_video_seg24s/test_file_1.mp4'
    output_audio_path = 'cstm/cstm_video_seg24s/test_file_1.wav'
    output_text_path = 'cstm/cstm_text_seg24s/test_file_1.txt'

    # Save the loaded video, audio, and associated text data
    save_vid_aud_txt(output_video_path, output_audio_path, output_text_path, video_data, audio_data, text, video_fps=25, audio_sample_rate=16000)
    ```

- An illustrative example of the label list file:

    ```
    cstm, cstm_video_seg24s/test_video_1.mp4, [input_length], [token_id]
    ```

    - The first part denotes the dataset (for example, `cstm`).

    - The second part specifies the relative path (`rel_path`) to the video or audio file within that dataset (for example, `cstm_video_seg24s/test_video_1.mp4`).

    - The third part indicates the number of frames in the video or the audio length divided by 640.

    - The final part gives the token ID (`token_id`), which is tokenized by the SentencePiece model (see Step 1). To transcribe into `token_id` from text, we provide [TextTransform.tokenize](./preparation/transforms.py) method. Please note that we do not include a comma for `[token_id]`. Therefore, you should concatenate all the string elements in the list to form a single string.

### Step 3. Building a dataset configuration file

Once you have pre-processed a custom dataset, the next step is to create a dataset configuration file, for instance, [cstm.yaml](./conf/data/dataset/cstm.yaml), which will connect the code with the dataset. In the configuration file, please make sure to specify the following parameters: `root`, `train_file`, `val_file`, and `test_file`. Assuimg that the training, validation and test label lists are located at `[root]/labels/[train_file]`, `[root]/labels/[val_file]` and `[root]/labels/[test_file]`, respectively.

- `root`: Path to the root directory where all preprocessed files are stored.

- `train_file`: Training file basename.

- `val_file`: Validation file basename.

- `test_file`: Testing file basename.

### Step 4. Training on the custom dataset

You can load our best available model for fine-tuning a VSR/ASR model. Checkpoints can be found at [model zoo](https://github.com/mpc001/auto_avsr#model-zoo).

```Shell
python main.py exp_dir=[exp_dir] \
               exp_name=[exp_name] \
               data.modality=[modality] \
               ckpt_path=[ckpt_path] \
               data/dataset=[dataset] \
               trainer.num_nodes=[num_nodes]
```

- You can set `data/dataset` to `cstm` to load `cstm` dataset for training and testing.
