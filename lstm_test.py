import torch

from torch.utils.data import Dataset
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import datasets
import time
import copy

from lr_finder import *
from helper import *

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data_dir = "data/train/bitmap"
num_epochs = 50

batch_size = 32


print("Initializing Datasets and Dataloaders...")

def bitmap_loader(path):
    with np.load(path) as data:
        data_len = data['arr_0'].shape[1]
        arr = np.pad(data['arr_0'], ((0, 0), (0, 22050-data_len)), 'constant')
        result = []
        for row in arr:
            unpacked_row = np.unpackbits(row)
            result.append(unpacked_row)

        return np.array(result)


bitmap_dataset = datasets.DatasetFolder(data_dir, loader=bitmap_loader, extensions='npz')

validation_split = 0.2
random_seed = 42

dataset_size = len(bitmap_dataset)
split = int(validation_split * dataset_size)

np.random.seed(random_seed)
indices = np.random.permutation(dataset_size)

train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = torch.utils.data.random_split(bitmap_dataset, [train_size, val_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dataloaders_dict = {}
dataloaders_dict['train'] = train_dataloader
dataloaders_dict['val'] = val_dataloader

# Neural Network

class RNN(nn.Module):
    def __init__(self, hidden_size=256, lstm_layers=2, cnn_start_channels=256):
        super(RNN, self).__init__()
        self.name = "CNN({})_LSTM({}_hidden_{})".format(cnn_start_channels, lstm_layers, cnn_start_channels)

        self.conv_layers = nn.Sequential(
            # input.size: 16x176400
            nn.Conv1d(in_channels=16, out_channels=cnn_start_channels, kernel_size=30, stride=10),
            # output: 64 x 17638
            nn.ReLU(),
            nn.BatchNorm1d(cnn_start_channels),
            # output 64x17638

            nn.Conv1d(in_channels=cnn_start_channels, out_channels=2*cnn_start_channels, kernel_size=30, stride=10),
            # output: 256 x 1762
            nn.ReLU(),
            nn.BatchNorm1d(2*cnn_start_channels),
            # output: 256 x 1762

            nn.Conv1d(in_channels=2*cnn_start_channels, out_channels=4*cnn_start_channels, kernel_size=30, stride=10),
            # output: 256 x 175
            nn.ReLU(),
            nn.BatchNorm1d(4*cnn_start_channels),
            # output: 256 x 175
        )

        self.rnn = nn.LSTM(input_size=4*cnn_start_channels,
                            hidden_size=hidden_size, dropout=0.2,
                            num_layers=lstm_layers)

        #self.rnn = nn.GRU(input_size=4*cnn_start_channels,
        #                    hidden_size=hidden_size, dropout=0.2,
        #                    num_layers=lstm_layers)

        self.fc = nn.Linear(hidden_size, 10)

    def forward(self, inputs, hidden):
        output = self.conv_layers(inputs)

        output = output.transpose(1, 2).transpose(0, 1)

        output = torch.tanh(output)
        output, hidden = self.rnn(output, hidden)

        output = self.fc(output[-1, :, :])

        return output, hidden

    def get_name(self):
        return self.name


def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25):
    since = time.time()

    val_acc_history = []
    val_loss_history = []

    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, hidden = model(inputs, None)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, best_epoch, (train_loss_history, val_loss_history, train_acc_history, val_acc_history)

model = RNN()
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.range_test(dataloaders_dict['train'], end_lr=100, num_iter=100, step_mode="exp")

#model, best_epoch, hist = train_model(model, dataloaders_dict, criterion, optimizer, scheduler=scheduler, num_epochs=num_epochs)


#model_filepath = './saved_models/{}_{:.0f}'.format(model.get_name(), 100*hist[3][best_epoch])

#plot_train_history(len(hist[0]), model_filepath, *hist)

#torch.save({
#    'epoch': best_epoch,
#    'model_state_dict': model.state_dict(),
#    'optimizer_state_dict': optimizer.state_dict(),
#    'loss': hist[1][best_epoch]
#}, model_filepath)
