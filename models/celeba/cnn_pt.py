import os
import torchvision.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models.baseline_constants import TENSORFLOW_OR_PYTORCH

assert TENSORFLOW_OR_PYTORCH == 'PT', f'TF_OR_PT indicates tensorflow or pytorch for nn models.' \
                                      f' Got TENSORFLOW_OR_PYTORCH={TENSORFLOW_OR_PYTORCH}'
IMAGES_DIR = os.path.join('..', 'data', 'celeba', 'data', 'raw', 'img_align_celeba')


class LeafPtDataset(Dataset):
    def __init__(self, list_IDs, labels):
        'Initialization'
        self._labels = labels
        self._list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self._list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self._list_IDs[index]

        # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        X = torchvision.io.read_image(os.path.join(IMAGES_DIR, ID)).float()
        y = self._labels[index]

        return X, y


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, img):
        x = self.conv(img)
        x = self.bn(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


class ClientModel(nn.Module):
    def __init__(self, seed, lr, num_classes):
        assert TENSORFLOW_OR_PYTORCH == 'PT', f'This class implements pytorch ClientModel.' \
                                              f' Got TENSORFLOW_OR_PYTORCH={TENSORFLOW_OR_PYTORCH}'
        super(ClientModel, self).__init__()
        self._size = -1.0
        self._lr = lr
        self._seed = seed
        self._val_set = None
        self._train_set = None
        self._test_loader = None
        self._train_loader = None
        self._val_loader = None
        self._criterion = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes

        # model
        self.block1 = Conv2DBlock(in_channels=3, out_channels=32)
        self.block2 = Conv2DBlock(in_channels=32, out_channels=32)
        self.block3 = Conv2DBlock(in_channels=32, out_channels=32)
        self.block4 = Conv2DBlock(in_channels=32, out_channels=32)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=4576, out_features=num_classes)

        self.initialize_weights()

        self.log_parameters()

        self._optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)

    def log_parameters(self):
        for (n, p) in self.named_parameters():
            print(f'parameter {n} numel {p.numel()} requires grad {p.requires_grad}')

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def set_params(self, model_params):
        for (p, mp) in zip(self.parameters(), model_params.parameters()):
            p.data = mp.data

    def get_params(self):
        return self.parameters()

    @property
    def size(self):
        if self._size <= 0.0:
            size_model = 0
            for param in self.parameters():
                if param.data.is_floating_point():
                    size_model += param.numel() * torch.finfo(param.data.dtype).bits
                else:
                    size_model += param.numel() * torch.iinfo(param.data.dtype).bits
            self._size = size_model
        return self._size

    def train_model(self, data, num_epochs=1, batch_size=10):
        dataset = LeafPtDataset(list_IDs=data['x'], labels=data['y'])
        self._train_set, self._val_set = torch.utils.data.random_split(dataset, lengths=[0.8, 0.2])
        self._train_loader = DataLoader(dataset=self._train_set, batch_size=batch_size, shuffle=True)
        self._val_loader = DataLoader(dataset=self._val_set, batch_size=batch_size, shuffle=False)

        for _ in range(num_epochs):
            self.run_epoch()

        update = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size

        return comp, update

    def run_epoch(self):
        self.train()
        epoch_loss, correct_counter, sample_counter = 0.0, 0, 0
        for data in tqdm(self._train_loader):
            imgs, labels = data
            sample_counter += labels.shape[0]
            self._optimizer.zero_grad()
            outputs = self(imgs)
            preds = outputs.max(1).indices
            correct = (preds == labels).sum().item()
            correct_counter += int(correct)
            loss = self._criterion(outputs, labels.long())
            loss.backward()
            self._optimizer.step()
            epoch_loss += float(loss)
            del loss, outputs, imgs, labels

        # results_dict = self.test()
        # val_acc = results_dict['accuracy']
        # val_loss = results_dict['loss']
        print(f'Train Epoch Accuracy {100 * correct_counter / sample_counter},'
              f' Train Epoch Loss {epoch_loss / sample_counter}')
        # print(f'Validation Epoch Accuracy {val_acc},'
        #       f' Validation Epoch Loss {val_loss}')

    @torch.no_grad()
    def test(self, data=None):
        self.eval()
        dataset = self._val_set if data is None else LeafPtDataset(list_IDs=data['x'], labels=data['y'])
        loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
        global_loss, correct_counter, sample_counter = 0.0, 0, 0
        for (imgs, labels) in tqdm(loader):
            sample_counter += labels.shape[0]
            outputs = self(imgs)
            loss = self._criterion(outputs, labels.long())
            preds = outputs.max(1).indices
            correct = (preds == labels).sum().item()
            correct_counter += int(correct)
            global_loss += float(loss)
        return {'accuracy': 100 * correct_counter / sample_counter, 'loss': global_loss / sample_counter}

    def save(self, path):
        torch.save({
            # 'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            # 'loss': loss,
        }, path)

    def close(self):
        pass
