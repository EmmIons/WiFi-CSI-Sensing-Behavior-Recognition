import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import time
torch.set_default_dtype(torch.float64)


# models
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(30 * 900, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        self.activ = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 30 * 900)
        x = self.fc(x)
        return self.activ(x)


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, ResBlock=Block, layer_list=[2, 2, 2, 2], num_classes=7):
        super(ResNet18, self).__init__()
        self.reshape = nn.Sequential(
            nn.ConvTranspose2d(1, 3, (12, 2), stride=(4, 1), padding=0, output_padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, kernel_size=(1, 3), stride=(1, 7)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)
        self.activ = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.reshape(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.activ(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


class RNN(nn.Module):
    def __init__(self,hidden_dim=64):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(30, hidden_dim,num_layers=1)
        self.fc = nn.Linear(hidden_dim, 7)
        self.activ = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 900, 30)
        x = x.permute(1, 0, 2)
        _, ht = self.rnn(x)
        outputs = self.activ(self.fc(ht[-1]))
        return outputs


class LSTM(nn.Module):
    def __init__(self,hidden_dim=64):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(30, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 7)
        self.activ = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 900, 30)
        x = x.permute(1, 0, 2)
        _, (ht,ct) = self.lstm(x)
        outputs = self.activ(self.fc(ht[-1]))
        return outputs


class CNN_GRU(nn.Module):
    def __init__(self):
        super(CNN_GRU,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(900, 900, 2, 2),
            nn.ReLU(True),
            nn.Conv1d(900, 900, 2, 2),
            nn.ReLU(True),
            nn.Conv1d(900, 900, 2, 1)
        )
        self.gru = nn.GRU(6, 128, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # batch x 1 x 250 x 90
        x = x.view(-1, 900, 30)
        x = self.encoder(x)
        # batch x 250 x 8
        x = x.permute(1, 0, 2)
        # 250 x batch x 8
        _, ht = self.gru(x)
        outputs = self.classifier(ht[-1])
        return outputs


# dataset
class WifiDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_dir = os.path.join(self.root_dir, 'data.pt')
        self.label_dir = os.path.join(self.root_dir, 'label.pt')
        self.data = torch.load(self.data_dir, map_location=device)
        self.label = torch.load(self.label_dir, map_location=device)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        label = torch.load(self.label_dir).to(device)
        return int(label.size(0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='CNN_GRU', choices=['MLP', 'ResNet18', 'RNN', 'LSTM', 'CNN_GRU'],
                        help='Choose model from:MLP, ResNet18, RNN, LSTM, CNN_GRU')
    parser.add_argument('--inference', action='store_true', help='Only inference')
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--batch_size', '-bs', type=int, default=128)
    parser.add_argument('--fromScratch', action='store_true', help='Train from scratch?')
    parser.add_argument('--gpu', default='cuda:0', help='choose which gpu to use')
    parser.add_argument('--checkpoint', '-ckp', default='./checkpoint/checkpoint.pth')
    args = parser.parse_args()

    # choose gpu
    device = args.gpu

    # train from scratch or checkpoint
    fromScratch = args.fromScratch

    # inference or train
    inference = args.inference

    # load data
    wifiDataset = WifiDataset(root_dir='./dataset_processed')

    # split train set and test set
    train_size = int(wifiDataset.__len__() * 0.9)
    val_size = wifiDataset.__len__() - train_size
    test_size = int(0.1 * train_size)
    train_dataset, val_dataset = torch.utils.data.random_split(wifiDataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(0))
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size - test_size, test_size],
                                                                generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # model
    if args.model == 'CNN_GRU':
        model = CNN_GRU().to(device)
    elif args.model == 'MLP':
        model = MLP().to(device)
    elif args.model == 'ResNet18':
        model = ResNet18().to(device)
    elif args.model == 'RNN':
        model = RNN().to(device)
    elif args.model == 'LSTM':
        model = LSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-6, weight_decay=1e-3)
    loss_func = nn.CrossEntropyLoss()

    # training
    # from scratch
    if fromScratch and not inference:
        print('--------Start training--------')
        for epoch in range(0, args.epoch):
            model.train()
            epoch_loss = 0
            epoch_accuracy = 0
            for _, (batch_x, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()
                predict_ys = model(batch_x)
                train_loss = loss_func(batch_y, predict_ys)
                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item() * batch_x.size()[0]
                epoch_accuracy += sum(row.all().int().item() for row in (predict_ys.ge(0.5) == batch_y))
            epoch_loss = epoch_loss / train_dataset.__len__()
            epoch_accuracy = epoch_accuracy / train_dataset.__len__()

            # validation
            model.eval()
            val_loss = 0
            val_accuracy = 0
            for _, (batch_x, batch_y) in enumerate(val_loader):
                predict_ys = model(batch_x)
                loss = loss_func(batch_y, predict_ys)
                val_loss += loss.item() * batch_x.size()[0]
                val_accuracy += sum(row.all().int().item() for row in (predict_ys.ge(0.5) == batch_y))
            val_loss = val_loss / val_dataset.__len__()
            val_accuracy = val_accuracy / val_dataset.__len__()
            print('Epoch: ', epoch, '| train loss: ', epoch_loss, '| train Accuracy: ', epoch_accuracy,
                  '| validation loss: ', val_loss, '| validation Accuracy: ', val_accuracy)

    # from checkpoint
    if not fromScratch:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        # train from checkpoint
        if not inference:
            while epoch < args.epoch:
                model.train()
                epoch_loss = loss
                for _, (batch_x, batch_y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    predict_y = model(batch_x)
                    train_loss = loss_func(batch_y, predict_y)
                    train_loss.backward()
                    optimizer.step()
                    epoch_loss += train_loss.item() * batch_x.size()[0]
                    epoch_accuracy += sum(row.all().int().item() for row in (predict_y.ge(0.5) == batch_y))
                epoch_loss = epoch_loss / train_dataset.__len__()
                epoch_accuracy = epoch_accuracy / train_dataset.__len__()

                # validation
                model.eval()
                val_loss = val_loss
                for _, (batch_x, batch_y) in enumerate(val_loader):
                    predict_y = model(batch_x)
                    loss = loss_func(batch_y, predict_y)
                    val_loss += loss.item() * batch_x.size()[0]
                    val_accuracy += sum(row.all().int().item() for row in (predict_y.ge(0.5) == batch_y))
                val_loss = val_loss / val_dataset.__len__()
                val_accuracy = val_accuracy / val_dataset.__len__()
                print('Epoch: ', epoch, '| train loss: ', epoch_loss, '| train Accuracy: ', epoch_accuracy,
                      '| validation loss: ', val_loss, '| validation Accuracy: ', val_accuracy)
                epoch += 1

    # save checkpoint
    if not inference:
        now = time.strftime("%m-%d-%H_%M_%S", time.localtime(time.time()))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': val_loss
        }, './checkpoint/checkpoint' + now + args.model + '.pth')

    # testing
    model.eval()
    test_loss = 0
    test_accuracy = 0
    predict_ys = []
    ground_truth = []
    for _, (batch_x, batch_y) in enumerate(test_loader):
        predict_y = model(batch_x)
        loss = loss_func(batch_y, predict_y)
        predict_ys.append(predict_y)
        ground_truth.append(batch_y)
        test_loss += loss.item() * batch_x.size()[0]
        test_accuracy += sum(row.all().int().item() for row in (predict_y.ge(0.5) == batch_y))
    test_loss = test_loss / test_dataset.__len__()
    test_accuracy = test_accuracy / test_dataset.__len__()
    print('--------test results--------')
    print('| Test loss:', test_loss, '| Test Accuracy:', test_accuracy, '| Ground Truth:', ground_truth, '| Predicts: ',
          predict_ys)
