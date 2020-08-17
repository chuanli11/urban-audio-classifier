# Lambda Notes

### Install

```
virtualenv -p /usr/bin/python3.6 venv

. venv/bin/activate

pip install tensorflow-gpu==1.15.3
pip install keras==2.3.1
pip install librosa
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
```

### Data Preparation

Download pre-generated spectrogram data

```
cd data
wget https://lambdalabs-files.s3-us-west-2.amazonaws.com/Navy/UrbanSound8K/X-mel_spec.npy
wget https://lambdalabs-files.s3-us-west-2.amazonaws.com/Navy/UrbanSound8K/y-mel_spec.npy
cd ..
```

Generate your own pre-processed data

Download [UrbanSound8K](https://urbansounddataset.weebly.com/download-urbansound8k.html) dataset to `urban-audio-classifier/UrbanSound8K`

```
pythons script_pre_processing.py
```

### Train and Evaluation

```
# With CPU
CUDA_VISIBLE_DEVICES= python demo_cnn.py

# With single GPU
python demo_cnn.py

# With multiple GPUs
python demo_cnn_multi_gpu.py
```

### Results

|   | CPU  | 1xGPU | 2xGPU |
|---|---|---|---|
| CNN (sec/epoch) | 60  | 2  | 1  |