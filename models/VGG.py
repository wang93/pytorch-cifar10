import torch.nn as nn
from SampleRateLearning.grad_scaling import Grad_Scaling

cfg = {
    'GAP': [],
    'VGG7S': [3, ],
    'VGG4S': [3, 'M', 6, 'M', 12, 'M', 12],
    'VGGSSS': [3, 'M', 6, 'M', 12, 'M', 24, 'M', 24, 'M'],
    'VGGSS': [4, 'M', 8, 'M', 16, 16, 'M', 32, 32, 'M', 32, 32, 'M'],
    'VGGS': [8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, class_num=10):
        super(VGG, self).__init__()
        num_features = 3
        self.features = self._make_layers(cfg[vgg_name])
        for s in reversed(cfg[vgg_name]):
            if isinstance(s, int):
                num_features = s
                break
        self.final_fc = nn.Linear(num_features, class_num, bias=False)

    def forward(self, x):
        out = self.features(x)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.final_fc(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def GAP(class_num=10):
    return VGG('GAP', class_num)


def VGG7S(class_num=10):
    return VGG('VGG7S', class_num)


def VGG4S(class_num=10):
    return VGG('VGG4S', class_num)


def VGGSSS(class_num=10):
    return VGG('VGGSSS', class_num)


def VGGSS(class_num=10):
    return VGG('VGGSS', class_num)


def VGGS(class_num=10):
    return VGG('VGGS', class_num)


def VGG11(class_num=10):
    return VGG('VGG11', class_num)


def VGG13(class_num=10):
    return VGG('VGG13', class_num)


def VGG16(class_num=10):
    return VGG('VGG16', class_num)


def VGG19(class_num=10):
    return VGG('VGG19', class_num)
