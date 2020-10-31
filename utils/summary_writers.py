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
# from SampleRateLearning.loss import SRL_BCELoss


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

    def record_iter(self, loss, global_step, pos_rate=None):
        if loss is not None:
            self.summary_writer.add_scalar('loss', loss.item(), global_step)
        if pos_rate is not None:
            self.summary_writer.add_scalar('pos_rate', pos_rate.item(), global_step)

    # def record(self, model, criterion, optimizer, loss=None, global_step=0):
    #     cur_lr = optimizer.param_groups[0]['lr']
    #     self.summary_writer.add_scalar('lr', cur_lr, global_step)
    #
    #     if loss is not None:
    #         self.summary_writer.add_scalar('loss', loss.item(), global_step)
    #
    #     final_layer = model.module.fc[-1].fc
    #     final_bias = final_layer.bias
    #     if final_bias is not None:
    #         self.summary_writer.add_scalar('final_bias', final_bias[-1].item(), global_step)
    #     weight_to_pos = final_layer.weight[-1, :]
    #     weight_shift_to_pos = weight_to_pos.mean().item()
    #     self.summary_writer.add_scalar('final_weight_shift',
    #                                    weight_shift_to_pos,
    #                                    global_step)
    #     self.summary_writer.add_scalar('relative_final_weight_shift',
    #                                    weight_shift_to_pos / weight_to_pos.abs().mean().item(),
    #                                    global_step)
    #
    #     final_w = model.module.braid.wlinear
    #     final_w_p = final_w.conv_p
    #     final_w_q = final_w.conv_q
    #     final_w_bias = final_w_p.bias
    #     final_wbn = model.module.braid.wbn.bn
    #     final_wbn_bias = final_wbn.bias
    #
    #     if final_w_bias is not None:
    #         final_w_bias = final_w_bias.data
    #         final_w_bias_avg = final_w_bias.mean().item()
    #         final_w_bias_max = final_w_bias.max().item()
    #         final_w_bias_min = final_w_bias.min().item()
    #         self.avg_summary_writer.add_scalar('final_w_bias', final_w_bias_avg, global_step)
    #         self.max_summary_writer.add_scalar('final_w_bias', final_w_bias_max, global_step)
    #         self.min_summary_writer.add_scalar('final_w_bias', final_w_bias_min, global_step)
    #
    #     final_w_weight_shift = torch.cat((final_w_p.weight.data, final_w_q.weight.data), dim=0).mean().item()
    #     self.summary_writer.add_scalar('final_w_weight_shift', final_w_weight_shift, global_step)
    #
    #     if final_wbn_bias is not None:
    #         final_wbn_bias = final_wbn_bias.data
    #         final_wbn_bias_avg = final_wbn_bias.mean().item()
    #         final_wbn_bias_max = final_wbn_bias.max().item()
    #         final_wbn_bias_min = final_wbn_bias.min().item()
    #         self.avg_summary_writer.add_scalar('final_wbn_bias', final_wbn_bias_avg, global_step)
    #         self.max_summary_writer.add_scalar('final_wbn_bias', final_wbn_bias_max, global_step)
    #         self.min_summary_writer.add_scalar('final_wbn_bias', final_wbn_bias_min, global_step)
    #
    #     final_wbn_weight = final_wbn.weight.data
    #     final_wbn_weight_avg = final_wbn_weight.mean().item()
    #     final_wbn_weight_max = final_wbn_weight.max().item()
    #     final_wbn_weight_min = final_wbn_weight.min().item()
    #     self.avg_summary_writer.add_scalar('final_wbn_weight', final_wbn_weight_avg, global_step)
    #     self.max_summary_writer.add_scalar('final_wbn_weight', final_wbn_weight_max, global_step)
    #     self.min_summary_writer.add_scalar('final_wbn_weight', final_wbn_weight_min, global_step)
    #
    #     if isinstance(criterion, SRL_BCELoss):
    #         self.summary_writer.add_scalar('pos_rate', criterion.sampler.pos_rate, global_step)
    #         if criterion.recent_losses is not None:
    #             self.pos_summary_writer.add_scalar('mean_loss', criterion.recent_losses[0], global_step)
    #             self.neg_summary_writer.add_scalar('mean_loss', criterion.recent_losses[1], global_step)
