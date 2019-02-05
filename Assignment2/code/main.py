import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from logger import Logger

class SoftmaxRegression(nn.Module):
    def __init__(self):
        super(SoftmaxRegression, self).__init__()
        self.fc = nn.Linear(784, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.25, training=self.training)
        x = x.view(-1, 64*12*12)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch, logger):
    model.train()
    loss_sum = 0
    loss_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        loss_count += 1
        if batch_idx % 10 == 0:
            print('Train Epoch: ' + str(epoch) + ' (' + str(round(100 * batch_idx / len(train_loader),2)) + '%) Loss: ' + str(loss.item()))

    info = { 'train_loss': loss_sum/loss_count }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

def test(model, device, test_loader, epoch, logger):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set - Average loss: ' + str(round(test_loss, 6)) + ', Accuracy: ' + str(round(100.0 * correct / len(test_loader.dataset),2)) + '%\n')

    info = { 'test_loss': test_loss, 'test_accuracy':  100.0 * correct / len(test_loader.dataset)}

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128, help='training batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate for Adadelta (default: 1.0)')
    parser.add_argument('--model', default='ConvNet', help='SoftmaxRegression, MLP or ConvNet (default: ConvNet)')
    parser.add_argument('--no-cuda', action='store_true', help='use CUDA or not')
    parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])), batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor()])), batch_size=1000, shuffle=False)

    if args.model == 'ConvNet':
        model = ConvNet().to(device)
    elif args.model == 'MLP':
        model = MLP().to(device)
    else:
        model = SoftmaxRegression().to(device)
        
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    logger = Logger('./logs_' + args.model)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, logger)
        test(model, device, test_loader, epoch, logger)

if __name__ == '__main__':
    main()
