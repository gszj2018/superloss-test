import math
import sys

import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST

DEVICE = torch.device("cuda:0")
writer = SummaryWriter()
torch.manual_seed(2)

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor()]
)

data_train = FashionMNIST('data',
                          train=True,
                          download=True,
                          transform=transform)

data_test = FashionMNIST('data',
                         train=False,
                         download=True,
                         transform=transform)

data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, pin_memory=True)
data_test_loader = DataLoader(data_test, batch_size=1024, pin_memory=True)


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10))
        ]))

    def forward(self, img):
        output = self.conv(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


model = LeNet5().to(DEVICE)


class SuperLoss(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.ra = None
        self.lam = lam

    def forward(self, loss: torch.Tensor):
        loss_detached = loss.detach().cpu()
        for v in loss_detached:
            self.add_history(v)
        tau = self.tau()
        log_sigma = self.log_sigma(loss_detached, tau)
        new_loss = torch.exp(log_sigma) * (loss - tau) + self.lam * log_sigma ** 2
        return new_loss.mean()

    def log_sigma(self, loss: torch.Tensor, tau):
        beta = (loss - tau) / self.lam
        inner = 0.5 * beta.clamp_min(-2.0 / math.e)
        lw = -scipy.special.lambertw(inner.numpy()).real.astype(np.float32)
        return torch.from_numpy(lw).to(DEVICE)

    def add_history(self, v):
        if self.ra is None:
            self.ra = v
        else:
            self.ra = self.ra * 0.9 + v * 0.1

    def tau(self):
        return self.ra


super_loss = SuperLoss(lam=1.0).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train(cur_epoch, use_super_loss):
    model.train()
    avg_loss = 0.0
    for image, label in data_train_loader:
        optimizer.zero_grad()
        out = model(image.to(DEVICE))
        loss = F.cross_entropy(out, label.to(DEVICE), reduction='none')
        avg_loss += loss.detach().cpu().sum()
        loss = super_loss(loss) if use_super_loss else loss.mean()
        loss.backward()
        optimizer.step()
    avg_loss /= len(data_train)
    print("Epoch {}:".format(cur_epoch))
    print("Mean training loss:{:.4f}".format(avg_loss))


best_acc = 0.0


def validate(cur_epoch):
    model.eval()
    avg_loss = 0.0
    correct_num = 0
    with torch.no_grad():
        for image, label in data_test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            out = model(image)
            loss = F.cross_entropy(out, label, reduction='none')
            avg_loss += loss.detach().cpu().sum()
            pred = out.detach().max(1)[1]
            correct_num += pred.eq(label.view_as(pred)).sum()
    avg_loss /= len(data_test)
    acc = float(correct_num) / len(data_test)
    global best_acc
    if acc > best_acc:
        best_acc = acc
    print("Mean validate loss:{:.4f}, Acc: {:.4f} (max {:.4f})".format(avg_loss, acc, best_acc))
    writer.add_scalar('acc', acc, cur_epoch)
    writer.flush()


def main(use_super_loss):
    for it in range(100):
        train(it, use_super_loss)
        validate(it)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError("Bad usage")
    sl = int(sys.argv[1]) > 0
    noise_rate = float(sys.argv[2])
    if noise_rate > 0.0:
        print("add noise to labels")
        data_train.targets = torch.tensor(data_train.targets)
        for i in range(len(data_train)):
            if torch.rand(()) < noise_rate:
                data_train.targets[i] = torch.randint(0, 10, ())
    print("Use SuperLoss: {}".format(sl))
    print("Noise rate: {}".format(noise_rate))
    main(sl)
