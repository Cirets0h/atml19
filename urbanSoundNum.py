import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import csv
import numpy as np
import librosa
import librosa.display
from utils_train import train, test


np.random.seed(123)
learning_rate = 0.005
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

train_transform = transforms.Compose([
    transforms.ToTensor()
])


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=88200, out_channels=16, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32)

        )
        self.conv_layer1 = nn.Sequential(

            nn.Conv1d(in_channels=88200, out_channels=512, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            #nn.BatchNorm1d(16),
            #nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.AdaptiveAvgPool1d((1))
            #nn.BatchNorm1d(32),
            #nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            # output: 64x56x56
            nn.ReLU(),
            #nn.BatchNorm1d(64)
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.BatchNorm1d(64),
            #nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv_layer5 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.BatchNorm1d(64)
            nn.AdaptiveAvgPool1d((1))

        )
        self.linear_layer = nn.Sequential(
            nn.Linear(64, 10)
        )

    def forward(self, input):
        #output = self.conv1d(input)

        output = self.conv_layer1(input)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = self.conv_layer4(output)
        output = self.conv_layer5(output)

        output = output.view(input.size(0), -1)
        output = self.linear_layer(output)
        return output



def csvToAudioList(filename,sourceDir): #interesting note: putting the data in a list and converting it to numpy is faster than directly putting it to an numpy array with genfromtext
    dataList = []
    with open(filename, "rt") as csvfile:
        lines = csv.reader(csvfile)
        dataList = list(lines)
        dataList.pop(0)
        #delete
        dataList = dataList[1500:2000]
        #

        audDataset = []
        labelDataset = []
        print(len(dataList))
        for x in dataList:
                audData, freq = librosa.load(sourceDir + x[0] + ".wav")
                #sound, sample_rate = torchaudio.load(sourceDir + x[0] + ".wav")
                if(len(audData) != 88200):
                    audData = fillWithZeros(audData)
                audDataset.append(audData)
                print(len(audDataset))
                labelDataset.append(labelTrans(x[1]))

    return audDataset, labelDataset

def fillWithZeros(audData):
    if(len(audData) < 88200):
        return np.append(audData,np.zeros((88200-len(audData),1),dtype=np.float32))
    else: #One dataset is longer
        print("AudioData longer than 88200")
        audData = audData[:88200]
        print(audData)
        print(len(audData))
        return audData


    return audData
def labelTrans(labelString):
    if(labelString == 'siren'):
        return 0
    elif(labelString == 'street_music'):
        return 1
    elif (labelString == 'drilling'):
        return 2
    elif (labelString == 'dog_bark'):
        return 3
    elif (labelString == 'children_playing'):
        return 4
    elif (labelString == 'gun_shot'):
        return 5
    elif (labelString == 'engine_idling'):
        return 6
    elif (labelString == 'air_conditioner'):
        return 7
    elif (labelString == 'jackhammer'):
        return 8
    elif (labelString == 'car_horn'):
        return 9



audList,labelList = csvToAudioList('/Users/manueldrazyk/Documents/Uni/FS19/ATML/Projekt/Proj/Data/urban-sound-classification/train/train.csv','/Users/manueldrazyk/Documents/Uni/FS19/ATML/Projekt/Proj/Data/urban-sound-classification/train/Train/')

split_refList = int(len(audList)*0.8)
train_audList, val_audList = audList[:split_refList], audList[split_refList:]
train_labelList, val_labelList = labelList[:split_refList], labelList[split_refList:]


class AudioDataset(Dataset):
    def __init__(self, data_audio, data_label, transform):

        self.data_set = np.array(data_audio)
        self.data_label1 = np.array(data_label)
        self.transform = transform

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        data_entry = self.data_set[index]
        #data_entry = self.transform(data_entry)
        data_entry = torch.from_numpy(data_entry).reshape(1,len(self.data_set[index]),1)
        #data_entry = data_entry.view(data_entry.size(0), 4)
        data_lab = torch.from_numpy(np.array([self.data_label1[index]]))

        return data_entry, data_lab

def fit(train_dataloader, val_dataloader, model, optimizer, loss_fn, n_epochs, scheduler=None):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    learning_rates = []
    for epoch in range(n_epochs):
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, loss_fn)
        val_loss, val_accuracy = test(model, val_dataloader, loss_fn)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        # We'll monitor learning rate -- just to show that it's decreasing
        learning_rates.append(optimizer.param_groups[0]['lr'])
        ########## Notify a scheduler that an epoch passed
        if scheduler:
            scheduler.step() # argument only needed for ReduceLROnPlateau
        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(epoch+1, n_epochs,
                                                                                                          train_losses[-1],
                                                                                                          train_accuracies[-1],
                                                                                                          val_losses[-1],
                                                                                                          val_accuracies[-1]))

    print(learning_rates)
    return train_losses, train_accuracies, val_losses, val_accuracies, learning_rates


trainDataset = AudioDataset(train_audList,train_labelList,train_transform)
valDataset = AudioDataset(val_audList,val_labelList, train_transform)

model = ConvNet()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
epochs = 25
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)
fit(trainDataset,valDataset,model,optimizer,loss_fn,epochs, scheduler )



