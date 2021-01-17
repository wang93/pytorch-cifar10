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
from SampleRateLearning import global_variables
from copy import deepcopy
from utils.lr_strategy_generator import MileStoneLR_WarmUp

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    parser = argparse.ArgumentParser(description="cifar with PyTorch")
    parser.add_argument('--dataset', default='cifar-10', type=str, help='dataset name')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--gamma', default=0.1, type=float, help='learning-rate-decay factor')
    parser.add_argument('--milestones', '-ms', default='[75,150]', type=str, help='milestones in lr schedule')
    parser.add_argument('--optim', default='adam', type=str, help='the optimizer for model')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--gpus', default='[0]', type=str, help='gpu devices to be used')
    parser.add_argument('--classes', '-c', default=None, type=str, help='classes to be considered')
    parser.add_argument('--sub_sample', '-s', default=None,
                        type=str, help='sub sample ratio of each class')
    parser.add_argument('--imbalance_rate', '-ir', default=None,
                        type=float, help='max(nums)/min(nums)')
    parser.add_argument('--exp', default='temp', type=str, help='experiment name')
    parser.add_argument('--arc', default='lenet', type=str, help='architecture name')
    parser.add_argument('--dtype', default='float', type=str, help='dtype of parameters and buffers')
    parser.add_argument('--seed', default=0, type=int, help='rand seed')
    parser.add_argument("--srl", action="store_true", help="sample rate learning or not.")
    parser.add_argument('--srl_lr', default=0.001, type=float, help='learning rate of srl')
    parser.add_argument('--srl_optim', default='adamw', type=str, help='the optimizer for srl')
    parser.add_argument("--srl_precision", '-ssp', action="store_true", help="srl according to soft precision")
    parser.add_argument('--sample_rates', default=None, type=str, help='sample rates in srl')
    parser.add_argument('--val_ratio', default=0., type=float, help='ratio of validation set in the training set')
    parser.add_argument('--valBatchSize', '-vb', default=16, type=int, help='validation batch size')
    parser.add_argument('--special_bn', default=-1, type=int, help='version of stable bn')
    parser.add_argument('--warmup_till', '-wt', default=1, type=int, help='version of stable bn')
    parser.add_argument('--warmup_mode', '-wm', default='const', type=str, help='version of stable bn')
    parser.add_argument("--weight_center", '-wc', action="store_true", help="centralize all the weights")
    parser.add_argument("--final_bn", default=-1., type=float, help='momentum of final bn')
    parser.add_argument("--final_zero", action="store_true", help="set params in the final layer to zero")
    args = parser.parse_args()

    if args.classes is None:
        if args.dataset == 'cifar-10':
            args.classes = list(range(10))
        elif args.dataset == 'cifar-100':
            args.classes = list(range(100))
        else:
            raise NotImplementedError
    else:
        args.classes = eval(args.classes)

    if args.sub_sample is None:
        if args.dataset == 'cifar-10':
            args.sub_sample = [1. for _ in range(10)]
        elif args.dataset == 'cifar-100':
            args.sub_sample = [1. for _ in range(100)]
        else:
            raise NotImplementedError
    else:
        args.sub_sample = eval(args.sub_sample)

    if args.imbalance_rate is not None:
        num_classes = len(args.classes)
        r = args.imbalance_rate ** (1/(1-num_classes))
        sub_sample = []
        for i in range(num_classes):
            sub_sample.append(r**i)
        args.sub_sample = sub_sample


    if args.srl and args.val_ratio <= 0.:
        args.srl_in_train = True

    if not args.srl:
        args.srl_alternate = False
        args.srl_in_train = False

    if args.sample_rates is not None:
        args.sample_rates = eval(args.sample_rates)

    args.milestones = eval(args.milestones)

    prepare_running(args)
    solver = Solver(args)
    solver.run()


class Solver(object):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.final_bn = None
        self.recorder = SummaryWriters(self.config, [CLASSES[c] for c in self.config.classes])
        global_variables.classes_num = len(self.config.classes)
        global_variables.train_batch_size = self.config.trainBatchSize

    @staticmethod
    def _sub_data(dataset, classes, ratios=None):
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

        return dataset

    @staticmethod
    def _split_data(dataset, test_trans, val_ratio=0.):
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
        val_set.transform = test_trans

        train_set.data = train_set.data[train_indices]
        train_set.targets = [train_set.targets[i] for i in train_indices]

        val_set.data = val_set.data[val_indices]
        val_set.targets = [val_set.targets[i] for i in val_indices]

        return train_set, val_set

    def load_data(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

        if self.config.dataset == 'cifar-10':
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        elif self.config.dataset == 'cifar-100':
            train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        else:
            raise NotImplementedError
        train_set = self._sub_data(train_set, self.config.classes, self.config.sub_sample)
        train_set, val_set = self._split_data(train_set, test_transform, self.config.val_ratio)

        if val_set is not None:
            from SampleRateLearning.sampler import ValidationBatchSampler
            batch_sampler = ValidationBatchSampler(data_source=val_set, batch_size=self.config.valBatchSize)
            self.val_loader = iter(torch.utils.data.DataLoader(dataset=val_set, batch_sampler=batch_sampler))

        if self.config.srl:
            from SampleRateLearning.sampler import SampleRateBatchSampler
            from SampleRateLearning.loss import SRL_CELoss as SRL_LOSS

            batch_sampler = SampleRateBatchSampler(data_source=train_set, batch_size=self.config.trainBatchSize)
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_sampler=batch_sampler)

            self.criterion = SRL_LOSS(sampler=batch_sampler,
                                      optim=self.config.srl_optim,
                                      lr=max(self.config.srl_lr, 0),
                                      sample_rates=self.config.sample_rates,
                                      precision_super=self.config.srl_precision,
                                      ).cuda()

        else:
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.config.trainBatchSize,
                                                            shuffle=True)
            self.criterion = nn.CrossEntropyLoss().cuda()

        if self.config.dataset == 'cifar-10':
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        elif self.config.dataset == 'cifar-100':
            test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        else:
            raise NotImplementedError
        self._sub_data(test_set, self.config.classes)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.config.testBatchSize, shuffle=False)

    def load_model(self):
        targets = self.train_loader.dataset.targets
        classes = list(set(targets))
        nums = [0 for _ in range(len(classes))]
        for t in targets:
            nums[t] += 1

        model_factory = {
            'lenet': LeNet,
            'gap': GAP,
            'vgg7s': VGG7S,
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
        model = model_factory[self.config.arc](class_num=len(self.config.classes))

        if self.config.special_bn >= 0:
            model_path = 'SampleRateLearning.special_batchnorm.batchnorm{0}'.format(self.config.special_bn)
            sbn = import_module(model_path)
            model = sbn.convert_model(model)

        self.model = nn.DataParallel(model).cuda()

        if self.config.weight_center:
            from WeightModification.recentralize import recentralize
            recentralize(self.model)

        if self.config.final_zero:
            final_fc = self.model.module.final_fc
            final_fc.weight.data = torch.zeros_like(final_fc.weight.data)
            if final_fc.bias is not None:
                final_fc.bias.data = torch.zeros_like(final_fc.bias.data)

        if self.config.final_bn > 0.:
            from SampleRateLearning.special_batchnorm.batchnorm41 import BatchNorm1d as final_bn1d
            self.final_bn = nn.DataParallel(final_bn1d(base_momentum=self.config.final_bn)).cuda()

        if self.config.optim == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        elif self.config.optim == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, weight_decay=0.0002)
        elif self.config.optim == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=0.)
        elif self.config.optim == 'adammw':
            from utils.optimizers import AdamMW
            self.optimizer = AdamMW(self.model.parameters(), lr=self.config.lr, weight_decay=0.)
        else:
            raise NotImplementedError

        self.scheduler = MileStoneLR_WarmUp(self.optimizer,
                                            milestones=self.config.milestones,
                                            gamma=self.config.gamma,
                                            warmup_till=self.config.warmup_till,
                                            warmup_mode=self.config.warmup_mode)

    def train(self, epoch):
        train_loss = 0
        train_correct = 0
        total = 0

        iter_num_per_epoch = len(self.train_loader)
        global_step = (epoch - 1) * iter_num_per_epoch

        for batch_num, (data, target) in enumerate(self.train_loader):
            if self.config.dtype == 'double':
                data, target = data.to(dtype=torch.double), target.to(dtype=torch.double)

            data, target = data.cuda(), target.cuda()
            global_variables.parse_target(target)

            # optimize model params
            self.model.train()
            if self.final_bn is not None:
                self.final_bn.train()
            self.criterion.eval()
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.final_bn is not None:
                output = self.final_bn(output)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # srl
            if self.config.srl:
                self.model.eval()
                if self.final_bn is not None:
                    self.final_bn.eval()
                self.criterion.train()
                val_data, val_target = self.val_loader.next()
                if self.config.dtype == 'double':
                    val_data, val_target = val_data.to(dtype=torch.double), val_target.to(dtype=torch.double)
                val_data, val_target = val_data.cuda(), val_target.cuda()
                with torch.no_grad():
                    val_output = self.model(val_data)
                    if self.final_bn is not None:
                        val_output = self.final_bn(val_output)
                self.criterion(val_output, val_target)

            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            train_correct += np.sum(prediction[1].cpu().numpy()
                                    == target.cpu().numpy())  # train_correct incremented by one if predicted right


            # record
            global_step += 1
            self.recorder.record_iter(loss, global_step,
                                      optimizer=self.optimizer,
                                      criterion=self.criterion)

        print('training loss:'.ljust(19) + '{:.5f}'.format(train_loss / (batch_num + 1)))
        if self.config.srl:
            m = lambda x: '{:.3f}'.format(x)
            print('sample rates:'.ljust(19) + ', '.join(map(m, self.criterion.sampler.sample_rates)))

        return train_loss, train_correct / total

    def test(self, epoch):
        self.model.eval()
        if self.final_bn is not None:
            self.final_bn.eval()
        if isinstance(self.criterion, nn.Module):
            self.criterion.eval()

        test_loss = 0
        test_correct = 0
        total = 0
        class_num = len(self.config.classes)
        cm = np.zeros((class_num, class_num), dtype=np.int)

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                if self.config.dtype == 'double':
                    data, target = data.to(dtype=torch.double), target.to(dtype=torch.double)
                data, target = data.cuda(), target.cuda()
                output = self.model(data)
                if self.final_bn is not None:
                    output = self.final_bn(output)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                _, prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction.cpu().numpy() == target.cpu().numpy())

                y_pred = prediction.view(-1).cpu().numpy().tolist()
                y_true = target.view(-1).cpu().numpy().tolist()
                cm += confusion_matrix(y_pred=y_pred, y_true=y_true, labels=list(range(len(self.config.classes))))

        accuracy = test_correct / total

        print('accuracy:'.ljust(19) + '{:.2f}%'.format(100. * accuracy))

        sample_nums = cm.sum(axis=1)
        hitted_nums = cm.diagonal()
        precisions = hitted_nums.astype(float) / sample_nums.astype(float)

        s = 'precisions:'.ljust(19)
        for p in precisions:
            s += '{:.1f}%, '.format(p * 100)
        print(s)

        average_precision = sum(precisions) / len(precisions)
        print('average precision:'.ljust(19) + '{:.2f}%'.format(average_precision * 100))

        worst_precision = min(precisions)

        print('worst precision:'.ljust(19) + '{:.1f}%'.format(worst_precision * 100))

        iter_num_per_epoch = len(self.train_loader)
        global_step = epoch * iter_num_per_epoch

        self.recorder.record_epoch(accuracy, precisions, global_step)

        return test_loss, test_correct / total, worst_precision

    def save(self):
        model_out_path = join('./exps', self.config.exp, "model.pth")
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        if self.config.dtype == 'double':
            torch.set_default_tensor_type(torch.DoubleTensor)
        self.load_data()
        self.load_model()

        accuracy = 0
        worst_precision = 0
        for epoch in range(1, self.config.epoch + 1):
            self.scheduler.step(epoch)

            if self.config.srl and self.config.srl_lr < 0:
                cur_lr = self.optimizer.param_groups[0]['lr']
                # cur_momentum = self.optimizer.param_groups[0]['momentum']
                self.criterion.optimizer.param_groups[0]['lr'] = cur_lr * 10.#/ (1. - cur_momentum)
            print("\n===> epoch: {0}/{1}".format(epoch, self.config.epoch))
            self.train(epoch)
            test_result = self.test(epoch)
            accuracy = max(accuracy, test_result[1])
            worst_precision = max(worst_precision, test_result[2])
            if epoch == self.config.epoch:
                print("\n===> BEST ACCURACY: %.2f%%" % (accuracy * 100))
                print("===> BEST WORST PRECISION: %.1f%%" % (worst_precision * 100))
                self.save()


if __name__ == '__main__':
    main()
