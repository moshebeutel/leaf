import numpy as np
import os

import torchvision.io
from baseline_constants import TENSORFLOW_OR_PYTORCH

if TENSORFLOW_OR_PYTORCH == 'TF':
    pass  # import tensorflow as tf
else:
    assert TENSORFLOW_OR_PYTORCH == 'PT', f'TF_OR_PT indicates tensorflow or pytorch for nn models.' \
                                          f' Got TENSORFLOW_OR_PYTORCH={TENSORFLOW_OR_PYTORCH}'
    import torch
    import torch.nn as nn
    from torchvision.datasets import CelebA
    from torch.utils.data import DataLoader, Dataset

from PIL import Image

from model import Model
from tqdm import trange, tqdm

IMAGE_SIZE = 84
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


class ClientModel_pt(nn.Module):
    def __init__(self, seed, lr, num_classes):
        super(ClientModel_pt, self).__init__()
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
        print('***')
        print(self)
        print('***')
        self.log_parameters()
        print('***')
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
        # print(x.shape)
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.block3(x)
        # print(x.shape)
        x = self.block4(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense(x)
        # print(x.shape)
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

    # @property
    # def criterion(self):
    #     if self._criterion is None:
    #         self._criterion = torch.nn.CrossEntropyLoss()
    # 
    #     return self._criterion
    # 
    # @property
    # def optimizer(self):
    #     """Optimizer to be used by the model."""
    #     if self._optimizer is None:
    #         self._optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
    # 
    #     return self._optimizer

    def train_model(self, data, num_epochs=1, batch_size=10):
        dataset = LeafPtDataset(list_IDs=data['x'], labels=data['y'])
        # dataset = CelebA(root='/home/user1/GIT/leaf_fork_pytorch/data/celeba', download=True, split='train')
        self._train_set, self._val_set = torch.utils.data.random_split(dataset, lengths=[0.8, 0.2])
        self._train_loader = DataLoader(dataset=self._train_set, batch_size=batch_size, shuffle=True)
        self._val_loader = DataLoader(dataset=self._val_set, batch_size=batch_size, shuffle=False)
        trange_pbar = tqdm(range(num_epochs), position=0)
        for epoch in trange_pbar:
            # train_loss, train_acc, val_loss, val_acc = self.run_epoch()
            train_loss, train_acc = self.run_epoch()
            trange_pbar.set_description(f'epoch {epoch} train_loss {train_loss}, train_acc {train_acc},')
                                        # f' val_loss {val_loss}, val_acc {val_acc}')

        # test_loss, test_acc = self.test(self._test_loader)
        # print(f'Test  test_loss {test_loss}, test_acc {test_acc}')

        update = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size
        return comp, update

    def run_epoch(self):
        self.train()
        epoch_loss, correct_counter, sample_counter = 0.0, 0, 0
        for data in tqdm(self._train_loader, position=1, leave=False):
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

        # results_dict = self.test(loader=self._val_loader)
        # val_acc = results_dict['accuracy']
        # val_loss = results_dict['loss']
        # return epoch_loss / sample_counter, 100 * correct_counter / sample_counter, val_loss, val_acc
        return 100 * correct_counter / sample_counter, epoch_loss / sample_counter

    @torch.no_grad()
    def test(self, data):
        self.eval()
        dataset = LeafPtDataset(list_IDs=data['x'], labels=data['y'])
        loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
        global_loss, correct_counter, sample_counter = 0.0, 0, 0
        for data in tqdm(loader):
            imgs, labels = data
            sample_counter += labels.shape[0]
            outputs = self(imgs)
            loss = self._criterion(outputs, labels.long())
            preds = outputs.max(1).indices
            correct = (preds == labels).sum().item()
            correct_counter += int(correct)
            global_loss += float(loss)
        return {'accuracy': 100 * correct_counter / sample_counter, 'loss': global_loss / sample_counter}

    def close(self):
        raise NotImplementedError

    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        raise NotImplementedError

    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        raise NotImplementedError

    def _load_image(self, img_name):
        raise NotImplementedError


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        input_ph = tf.placeholder(
            tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
        out = input_ph
        for _ in range(4):
            out = tf.layers.conv2d(out, 32, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        logits = tf.layers.dense(out, self.num_classes)
        label_ph = tf.placeholder(tf.int64, shape=(None,))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_ph,
            logits=logits)
        predictions = tf.argmax(logits, axis=-1)
        minimize_op = self.optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(
            tf.equal(label_ph, tf.argmax(input=logits, axis=1)))
        return input_ph, label_ph, minimize_op, eval_metric_ops, tf.math.reduce_mean(loss)

    def process_x(self, raw_x_batch):
        x_batch = [self._load_image(i) for i in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        return raw_y_batch

    def _load_image(self, img_name):
        img = Image.open(os.path.join(IMAGES_DIR, img_name))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
        return np.array(img)
