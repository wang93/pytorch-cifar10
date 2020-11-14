import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import transforms as transforms
import argparse
from models import *
from sklearn.metrics import confusion_matrix
import numpy as np
from os.path import join
from importlib import import_module
import random
from utils.standard_actions import prepare_running
from utils.summary_writers import SummaryWriters
from SampleRateLearning.stable_batchnorm import global_variables
from copy import deepcopy

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--optim', default='adamw', type=str, help='the optimizer for model')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--gpus', default='[0]', type=str, help='gpu devices to be used')
    parser.add_argument('--classes', '-c', default='[0,1,2,3,4,5,6,7,8,9]', type=str, help='classes to be considered')
    parser.add_argument('--sub_sample', '-s', default='[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]',
                        type=str, help='sub sample ratio of each class')
    parser.add_argument('--exp', default='temp', type=str, help='experiment name')
    parser.add_argument('--arc', default='lenet', type=str, help='architecture name')
    parser.add_argument('--seed', default=0, type=int, help='rand seed')
    parser.add_argument("--srl", action="store_true", help="sample rate learning or not.")
    parser.add_argument('--srl_lr', default=0.001, type=float, help='learning rate of srl')
    parser.add_argument('--srl_optim', default='adamw', type=str, help='the optimizer for srl')
    parser.add_argument('--srl_norm', action="store_true", help="use normed srl")
    parser.add_argument("--srl_in_train", '-st', action="store_true", help="sample rate learning in the training set")
    parser.add_argument('--pos_rate', default=None, type=float, help='pos_rate in srl')
    parser.add_argument('--val_ratio', default=0., type=float, help='ratio of validation set in the training set')
    parser.add_argument('--valBatchSize', '-vb', default=16, type=int, help='validation batch size')
    parser.add_argument("--sri", action="store_true", help="sample rate inference or not.")
    parser.add_argument('--stable_bn', default=-1, type=int, help='version of stable bn')
    args = parser.parse_args()

    if args.srl and args.val_ratio <= 0.:
        args.srl_in_train = True

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
        self.val_batch_size = config.valBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.classes = eval(config.classes)
        self.ratios = eval(config.sub_sample)
        self.srl = config.srl
        self.srl_lr = config.srl_lr
        self.val_ratio = config.val_ratio
        self.sri = config.sri
        self.recorder = SummaryWriters(config, [CLASSES[c] for c in self.classes])
        self.stable_bn = config.stable_bn
        self.config = config

        global_variables.classes_num = len(self.classes)

        if self.srl and len(self.classes) != 2:
            raise NotImplementedError

    @staticmethod
    def _sub_data(dataset, classes, ratios=None, val_ratio=0.):
        chosen_indices = []
        if ratios is None:
            ratios = [1., ] * len(classes)

        for c, r in zip(classes, ratios):
            indices = [i for i, l in enumerate(dataset.targets) if l == c]
            num = len(indices)
            sub_num = round(num * r)
            indices = random.sample(indices, sub_num)
            chosen_indices.extend(indices)

        dataset.data = dataset.data[chosen_indices]
        dataset.targets = [classes.index(dataset.targets[i]) for i in chosen_indices]

        dataset.classes = [dataset.classes[c] for c in classes]
        dataset.class_to_idx = {c: i for i, c in enumerate(dataset.classes)}

        if val_ratio <= 0.:
            return dataset, None

        train_indices = []
        val_indices = []

        for c in range(len(dataset.classes)):
            indices = [i for i, l in enumerate(dataset.targets) if l == c]
            num = len(indices)
            sub_num = round(num * val_ratio)
            v_indices = random.sample(indices, sub_num)
            t_indices = list(set(indices) - set(v_indices))
            val_indices.extend(v_indices)
            train_indices.extend(t_indices)

        train_set = dataset
        val_set = deepcopy(dataset)

        train_set.data = train_set.data[train_indices]
        train_set.targets = [train_set.targets[i] for i in train_indices]

        val_set.data = val_set.data[val_indices]
        val_set.targets = [val_set.targets[i] for i in val_indices]

        return train_set, val_set

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        train_set, val_set = self._sub_data(train_set, self.classes, self.ratios, self.val_ratio)

        if val_set is not None:
            from SampleRateLearning.sampler import ValidationBatchSampler
            batch_sampler = ValidationBatchSampler(data_source=val_set, batch_size=self.val_batch_size)
            self.val_loader = iter(torch.utils.data.DataLoader(dataset=val_set, batch_sampler=batch_sampler))

        if self.srl or self.sri:
            from SampleRateLearning.sampler import SampleRateBatchSampler
            batch_sampler = SampleRateBatchSampler(data_source=train_set, batch_size=self.train_batch_size)
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_sampler=batch_sampler)

            if self.srl:
                from SampleRateLearning.loss import SRL_CELoss
                self.criterion = SRL_CELoss(sampler=batch_sampler,
                                            optim=self.config.srl_optim,
                                            lr=max(self.srl_lr, 0),
                                            pos_rate=self.config.pos_rate,
                                            in_train=self.config.srl_in_train,
                                            norm=self.config.srl_norm
                                            ).cuda()
            else:
                from SampleRateLearning.loss import SRI_CELoss
                self.criterion = SRI_CELoss(sampler=batch_sampler).cuda()

        else:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size,
                                                            shuffle=True)
            self.criterion = nn.CrossEntropyLoss().cuda()

        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self._sub_data(test_set, self.classes)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        model_factory = {
            'lenet': LeNet,
            'gap': GAP,
            'vgg4s': VGG4S,
            'vggsss': VGGSSS,
            'vggss': VGGSS,
            'vggs': VGGS,
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
            'wresnet': WideResNet
        }
        model = model_factory[self.arc](class_num=len(self.classes))

        if self.stable_bn >= 0:
            model_path = 'SampleRateLearning.stable_batchnorm.batchnorm{0}'.format(self.stable_bn)
            sbn = import_module(model_path)
            model = sbn.convert_model(model)

        self.model = nn.DataParallel(model).cuda()
        if self.config.optim == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.config.optim == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.)
        elif self.config.optim == 'adammw':
            from utils.optimizers import AdamMW
            self.optimizer = AdamMW(self.model.parameters(), lr=self.lr, weight_decay=0.)
        else:
            raise NotImplementedError

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        iter_num_per_epoch = len(self.train_loader)
        global_step = (epoch - 1) * iter_num_per_epoch

        for batch_num, (data, target) in enumerate(self.train_loader):
            if self.val_loader is not None:
                val_data, val_target = self.val_loader.next()
                data = torch.cat((data, val_data), dim=0)
                target = torch.cat((target, val_target), dim=0)

            data, target = data.cuda(), target.cuda()
            global_variables.parse_target(target)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            global_step += 1
            if self.srl or self.sri:
                pos_rate = self.criterion.pos_rate
            else:
                pos_rate = None
            self.recorder.record_iter(loss, global_step,
                                      pos_rate=pos_rate,
                                      optimizer=self.optimizer,
                                      criterion=self.criterion)

            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            # progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        print('training loss: {:.5f}'.format(train_loss / (batch_num + 1)))
        if self.srl or self.sri:
            print('pos rate: {:.4f}'.format(self.criterion.pos_rate))

        return train_loss, train_correct / total

    def test(self, epoch):
        # print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        class_num = len(self.classes)
        cm = np.zeros((class_num, class_num), dtype=np.int)

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.cuda(), target.cuda()
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                _, prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction.cpu().numpy() == target.cpu().numpy())

                # progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

                y_pred = prediction.view(-1).cpu().numpy().tolist()
                y_true = target.view(-1).cpu().numpy().tolist()
                cm += confusion_matrix(y_pred=y_pred, y_true=y_true, labels=list(range(len(self.classes))))

        accuracy = test_correct / total

        print('test accuracy: {:.2f}%'.format(100. * accuracy))

        # print('confusion matrix:')
        # pprint(cm)

        sample_nums = cm.sum(axis=1)
        hitted_nums = cm.diagonal()
        precisions = hitted_nums.astype(float) / sample_nums.astype(float)

        s = 'the precisions are: '
        for p in precisions:
            s += '{:.1f}%, '.format(p * 100)
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
            if self.srl and self.srl_lr < 0:
                cur_lr = self.optimizer.param_groups[0]['lr']
                self.criterion.optimizer.param_groups[0]['lr'] = cur_lr
                #     self.summary_writer.add_scalar('lr', cur_lr, global_step)
            print("\n===> epoch: %d/200" % epoch)
            self.train(epoch)
            test_result = self.test(epoch)
            accuracy = max(accuracy, test_result[1])
            worst_precision = max(worst_precision, test_result[2])
            if epoch == self.epochs:
                print("\n===> BEST ACCURACY: %.2f%%" % (accuracy * 100))
                print("===> BEST WORST PRECISION: %.1f%%" % (worst_precision * 100))
                self.save()


if __name__ == '__main__':
    main()
