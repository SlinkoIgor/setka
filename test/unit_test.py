from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import keras
import random

import numpy

import torch

import scorch.base
import scorch.callbacks
#import scorch.scripts

numpy.random.seed(0)
random.seed(0)

# Computing result with keras

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(list(x_train.shape) + [1,])
x_test = x_test.reshape(list(x_test.shape) + [1,])

train_reorder = numpy.random.permutation(len(x_train))

x_train = x_train[train_reorder]
y_train = y_train[train_reorder]

n_valid = int(len(x_train) * 0.2)

x_valid = x_train[:n_valid]
y_valid = y_train[:n_valid]

x_train = x_train[n_valid:]
y_train = y_train[n_valid:]

labels_train = numpy.zeros([len(y_train), 10])
labels_valid = numpy.zeros([len(y_valid), 10])
labels_test = numpy.zeros([len(y_test), 10])

labels_train[numpy.arange(len(y_train)), y_train] = 1
labels_valid[numpy.arange(len(y_valid)), y_valid] = 1
labels_test[numpy.arange(len(y_test)), y_test] = 1


# Computing result with scorch

class DataSet(scorch.base.DataSet):
    def __init__(self):
        super(DataSet, self).__init__()
        self.data = {
            'train': torch.from_numpy(x_train).transpose(1, 3).float(),
            'valid': torch.from_numpy(x_valid).transpose(1, 3).float(),
            'test':  torch.from_numpy(x_test).transpose(1, 3).float()
        }

        self.labels = {
            'train': torch.from_numpy(y_train).long(),
            'valid': torch.from_numpy(y_valid).long(),
            'test' : torch.from_numpy(y_test).long()
        }


class Network(scorch.base.Network):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, 5, padding=2)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(64, 128, 5, padding=2)
        self.pool2 = torch.nn.MaxPool2d(2)

        self.fc1 = torch.nn.Linear(6272, 1024)
        self.fc2 = torch.nn.Linear(1024, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, input):
        res = input[0]
        res = self.pool1(torch.relu(self.conv1(res)))
        res = self.pool2(torch.relu(self.conv2(res)))

        res = res.view([res.size(0), -1])

        res = torch.relu(self.fc1(res))
        res = torch.relu(self.fc2(res))
        res = self.fc3(res)

        return [res]

net = Network()

def criterion(input, target):
    return torch.nn.functional.cross_entropy(input[0], target[0])

def accuracy(preds, target):
    return (preds[0].argmax(dim=1) == target[0]).float().mean()

trainer = scorch.base.Trainer(net,
                  criterion=criterion,
                  optimizers=[
                    scorch.base.OptimizerSwitch(
                    net, torch.optim.Adam, lr=3.0e-5)],
                  callbacks=[scorch.callbacks.ComputeMetrics(
                                metrics={'main': accuracy, 'loss': criterion}),
                             scorch.callbacks.MakeCheckpoints(),
                             scorch.callbacks.SaveResult(),
                             scorch.callbacks.WriteToTensorboard()],
                  seed=1,
                  silent=False)
dataset = DataSet()

for epoch in range(1):
    trainer.train_one_epoch(dataset,
                            num_workers=2,
                            max_iterations=20,
                            batch_size=32)

    trainer.validate_one_epoch(dataset,
                               subset='valid',
                               num_workers=2,
                               max_iterations=20,
                               batch_size=32)




res = trainer.train(
             dataset,
             batch_size=32,
             num_workers=2,
             validate_on_train=True,
             max_train_iterations=20,
             max_valid_iterations=20,
             max_test_iterations=2,
             solo_test=True,
             epochs=5)

trainer.predict(dataset,
                batch_size=32,
                max_iterations=2)

trainer.predict(dataset,
                subset='valid',
                batch_size=32,
                max_iterations=2)

#trainer.save('./checkpoints/checkpoint')

#trainer.load('./checkpoints/checkpoint.pth.tar')