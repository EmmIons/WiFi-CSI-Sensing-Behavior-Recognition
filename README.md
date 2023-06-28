# WiFi-CSI-Sensing-Behavior-Recognition
Train CNN, MLP, ResNet18, RNN, LSTM on CSI data to classify behavior.

# Requirements
`python==3.11.3`
`torch==2.0.0+cu117`
`torchvision==0.15.1+cu117`
`pandas==2.0.2`

# Dataset
CSI data of 30*900 every sample,there exist 600 samples.You can change to your own dataset by modifing and runing dataset_process.py.
Dataset using here is not offered!
# Run
'python main.py --model MLP --epoch 50 --gpu cuda:0 --batch_size 128'
for '--model', you can choose from 'MLP, ResNet18, RNN, LSTM, CNN_GRU'.
if you want to train model from scratch, add '--fromScratch' parameter, for example:
'python main.py --model MLP --epoch 50 --gpu cuda:0 --batch_size 128 --fromScratch'
or using follow to train a model from checkpoint:
'python main.py --model MLP --epoch 50 --gpu cuda:0 --batch_size 128 --checkpoint ./checkpoint/checkpoint.pth'
# Inference
Only inference using saved checkpoint:
'python main.py --model MLP --checkpoint ./checkpoint/checkpoint.pth'
