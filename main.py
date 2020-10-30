import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms

import argparse

from models import *

from sklearn.metrics import confusion_matrix
import numpy as np
from os.path import join

from utils.standard_actions import prepare_running
from utils.summary_writers import SummaryWriters

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--gpus', default='[0]', type=str, help='gpu devices to be used')
    parser.add_argument('--classes', '-c', default='[0,1,2,3,4,5,6,7,8,9]', type=str, help='classes to be considered')
    parser.add_argument('--exp', default='temp', type=str, help='experiment name')
    parser.add_argument('--arc', default='lenet', type=str, help='architecture name')
    parser.add_argument('--seed', default=0, type=int, help='rand seed')
    args = parser.parse_args()

    prepare_running(args)

    solver = Solver(args)
    solver.run()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.arc = config.arc
        self.epochs = config.epoch
        self.exp = config.exp
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.recorder = SummaryWriters(config, CLASSES)
        self.classes = eval(config.classes)

    @staticmethod
    def _sub_data(dataset, classes):
        indices = [i for i, l in enumerate(dataset.targets) if l in classes]
        dataset.data = dataset.data[indices]
        dataset.targets = [dataset.targets[i] for i in indices]

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        self._sub_data(train_set, self.classes)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)

        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self._sub_data(test_set, self.classes)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        model_factory = {
            'lenet': LeNet,
            'vgg11': VGG11,
            'vgg16': VGG16,
            'vgg19': VGG19,
            'googlenet': GoogLeNet,
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152,
            'densenet121': DenseNet121,
            'densenet161': DenseNet161,
            'densenet169': DenseNet169,
            'densenet201': DenseNet201,
            'WideResNet': WideResNet
        }
        self.model = model_factory[self.arc]().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        iter_num_per_epoch = len(self.train_loader)
        global_step = (epoch-1) * iter_num_per_epoch

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            global_step += 1
            self.recorder.record_iter(loss, global_step)

            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            # progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        print('training loss: {:.5f}'.format(train_loss / (batch_num + 1)))

        return train_loss, train_correct / total

    def test(self, epoch):
        # print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        class_num = len(CLASSES)
        cm = np.zeros((class_num, class_num), dtype=np.int)

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                _, prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction.cpu().numpy() == target.cpu().numpy())

                # progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

                cm += confusion_matrix(y_pred=prediction.view(-1).cpu().numpy(), y_true=target.view(-1).cpu().numpy())

        accuracy = test_correct / total

        print('test accuracy: {:.2f}%'.format(100. * accuracy))

        # print('confusion matrix:')
        # pprint(cm)

        sample_nums = cm.sum(axis=1)
        hitted_nums = cm.diagonal()
        precisions = hitted_nums.astype(float) / sample_nums.astype(float)

        s = 'the precisions are: '
        for p in precisions:
            s += '{:.1f}%, '.format(p*100)
        print(s)

        worst_precision = min(precisions)

        print('the worst precision is {:.1f}%'.format(worst_precision * 100))

        iter_num_per_epoch = len(self.train_loader)
        global_step = epoch * iter_num_per_epoch

        self.recorder.record_epoch(accuracy, precisions, global_step)

        return test_loss, test_correct / total, worst_precision

    def save(self):
        model_out_path = join('./exps', self.exp, "model.pth")
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        worst_precision = 0
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            print("\n===> epoch: %d/200" % epoch)
            self.train(epoch)
            test_result = self.test(epoch)
            accuracy = max(accuracy, test_result[1])
            worst_precision = max(worst_precision, test_result[2])
            if epoch == self.epochs:
                print("===> BEST ACCURACY: %.2f%%" % (accuracy * 100))
                print("===> BEST WORST PRECISION: %.1f%%" % (worst_precision * 100))
                self.save()


if __name__ == '__main__':
    main()
