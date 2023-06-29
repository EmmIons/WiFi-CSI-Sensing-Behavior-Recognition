# WiFi-CSI-Sensing-Behavior-Recognition
Train CNN, MLP, ResNet18, RNN, LSTM on CSI data to classify 7 behaviors.`walk, stand, sit, run, fall, bow, null`

# Requirements
`python==3.11.3`<br />
`torch==2.0.0+cu117`<br />
`pandas==2.0.2`<br />

# Dataset
CSI data of 30*900 every sample,there exist 600 samples.You can change to your own dataset by modifing `dataset_process.py` and `main.WifiDataset()`.<br />
Dataset used here is not offered!
# Run
`python main.py --model MLP --epoch 50 --gpu cuda:0 --batch_size 128`<br />
For `--model`, you can choose from `MLP, ResNet18, RNN, LSTM, CNN_GRU`.<br />
<br />
If you want to train model from scratch, add `--fromScratch` parameter, for example:<br />
`python main.py --model MLP --epoch 50 --gpu cuda:0 --batch_size 128 --fromScratch`
<br />
Or using follow to train a model from checkpoint:<br />
`python main.py --model MLP --epoch 50 --gpu cuda:0 --batch_size 128 --checkpoint ./checkpoint/checkpoint.pth`<br />
# Inference
Only inference using saved checkpoint:<br />
`python main.py --model MLP --checkpoint ./checkpoint/checkpoint.pth`<br />
