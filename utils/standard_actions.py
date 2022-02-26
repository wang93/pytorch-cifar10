# encoding: utf-8
import os
import random
import subprocess
import sys

import numpy as np
import torch

from utils.serialization import Logger
from pprint import pprint


def _random_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def prepare_running(opt):
    print(torch.cuda.is_available())
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in eval(opt.gpus)])
    sys.stdout = Logger(os.path.join('./exps', opt.exp, 'log_train.txt'))
    print('current commit hash: {}'.format(subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()))
    pprint(opt)
    _random_seed(opt.seed)
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
