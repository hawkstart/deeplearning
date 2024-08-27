from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import visdom
import onnx
import torch.nn.functional as F







# 自定义损失函数


class MyCustomLoss(nn.Module):
    def __init__(self):
        super(MyCustomLoss, self).__init__()

    def forward(self, output, labels, epoch):
        # 将标签转换为热编码
        labels= F.one_hot(labels, num_classes=10)
        # 计算损失函数
        evidence_vector=output+1.0
        S=evidence_vector.sum(dim=1)
        S_temp=S+1
        P=evidence_vector/S[:,None]
        L1=(labels-P)**2+P*(1-P)/S_temp[:,None]

        # 计算正则项
        a=labels+(1.0-labels)*evidence_vector
        a_addrow=a.sum(dim=1)
        a_addrow_digamma=torch.special.digamma(a_addrow)
        L2_2_temp=(a-1.0)*(torch.special.digamma(a)-a_addrow_digamma[:,None])
        L2_2=L2_2_temp.sum(dim=1)


        a_addrow_gamma=torch.special.gamma(a_addrow)
        a_gamma=torch.special.gamma(a)
        a_gamma_prod=torch.prod(a_gamma, dim=1)
        L2_1_temp1=a_addrow_gamma/torch.special.gamma(10)
        L2_1_temp2=L2_1_temp1/a_gamma_prod
        L2_1=torch.log(L2_1_temp2)
        L2=L2_1+L2_2

        # 参数lamda
        t=epoch/10
        lamda=torch.min(1.0,t)
        loss_temp=L1+lamda*L2
        loss=torch.sum(loss_temp)


        return loss




















viz = visdom.Visdom()

data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

net = LeNet5()
criterion = MyCustomLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}


def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels, epoch)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        # Update Visualization
        if viz.check_connection():
            cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
                                     win=cur_batch_win, name='current_batch_loss',
                                     update=(None if cur_batch_win is None else 'replace'),
                                     opts=cur_batch_win_opts)

        loss.backward()
        optimizer.step()


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


def train_and_test(epoch):
    train(epoch)
    test()

    dummy_input = torch.randn(1, 1, 32, 32, requires_grad=True)
    torch.onnx.export(net, dummy_input, "lenet.onnx")

    onnx_model = onnx.load("lenet.onnx")
    onnx.checker.check_model(onnx_model)


def main():
    for e in range(1, 16):
        train_and_test(e)


if __name__ == '__main__':
    main()
