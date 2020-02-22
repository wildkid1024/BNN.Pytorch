# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Function

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from BinaryNet import BWN
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("=>using GPU")
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
else:
    print("=>using CPU")
    
net = BWN(28*28, 10)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

train_dataset = datasets.MNIST(
    root='./dataset', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(
    root='./dataset', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
   
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs = inputs.view(-1, 28*28)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()              
        for p in list(net.parameters()): 
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(net.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 20 == 0:
            print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)' 
            % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


best_acc = 0.0
start_epoch = 0

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(-1, 28*28)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 20 == 0:
                print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/BNN-MNIST-pretrain.pt')
        best_acc = acc
        
def load_model():
    global start_epoch
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/BNN-MNIST-pretrain.pt')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
#     print(sta)
    
def print_weight():
    load_model()
    paras = net.named_parameters()
    for name,w in paras:
        print(name, w)

def main():
    global start_epoch
    # load_model()
    for epoch in range(start_epoch, start_epoch + 10):
        train(epoch)
        test(epoch)
    
    print_weight()

if __name__ == "__main__":
    main()