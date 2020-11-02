# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/10/16 15:30

"""
SummaryWriters for braidosnets
"""

from tensorboardX import SummaryWriter
import torch
from os.path import join as pjoin
from SampleRateLearning.loss import SRL_BCELoss, SRI_BCELoss


class SummaryWriters(object):
    def __init__(self, opt, classes):
        exp_dir = pjoin('./exps', opt.exp, 'tensorboard_log')
        self.summary_writer = SummaryWriter(pjoin(exp_dir, 'common'))
        self.max_summary_writer = SummaryWriter(pjoin(exp_dir, 'max'))
        self.min_summary_writer = SummaryWriter(pjoin(exp_dir, 'min'))
        self.avg_summary_writer = SummaryWriter(pjoin(exp_dir, 'avg'))

        self.class_summary_writers = []
        class_num = len(classes)
        for i in range(class_num):
            self.class_summary_writers.append(SummaryWriter(pjoin(exp_dir, str(i))))

    def record_epoch(self, acc, precisions, global_step):
        self.summary_writer.add_scalar('accuracy', acc, global_step)
        self.summary_writer.add_scalar('worst precision', min(precisions), global_step)
        for writer, precision in zip(self.class_summary_writers, precisions):
            writer.add_scalar('precision', precision, global_step)

    def record_iter(self, loss, global_step, pos_rate=None, optimizer=None, criterion=None):
        if loss is not None:
            self.summary_writer.add_scalar('loss', loss.item(), global_step)

        if isinstance(criterion, (SRI_BCELoss, SRL_BCELoss)):
            for writer, c_loss in zip(self.class_summary_writers, criterion.recent_losses):
                writer.add_scalar('classwise_loss', loss.item(), global_step)

        if pos_rate is not None:
            if isinstance(pos_rate, torch.Tensor):
                pos_rate = pos_rate.cpu().item()
            self.summary_writer.add_scalar('pos_rate', pos_rate, global_step)
        if optimizer is not None:
            cur_lr = optimizer.param_groups[0]['lr']
            self.summary_writer.add_scalar('lr', cur_lr, global_step)
