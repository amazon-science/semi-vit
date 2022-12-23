# ------------------------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Licensed under the CC-BY-NC 4.0 license.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from timm.models.layers import DropPath

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader_x: Iterable, data_loader_u: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, pseudo_mixup_fn=None,
                    log_writer=None, args=None):
    model.train(True)
    model_ema.ema.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    epoch_x = epoch * math.ceil(len(data_loader_u) / len(data_loader_x))
    if args.distributed:
        print("set epoch={} for labeled sampler".format(epoch_x))
        data_loader_x.sampler.set_epoch(epoch_x)
        print("set epoch={} for unlabeled sampler".format(epoch))
        data_loader_u.sampler.set_epoch(epoch)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    data_iter_x = iter(data_loader_x)
    for data_iter_step, (samples_u, targets_u) in enumerate(metric_logger.log_every(data_loader_u, args.print_freq, header)):
        try:
            samples_x, targets_x = next(data_iter_x)
        except Exception:
            epoch_x += 1
            print("reshuffle data_loader_x at epoch={}".format(epoch_x))
            if args.distributed:
                print("set epoch={} for labeled sampler".format(epoch_x))
                data_loader_x.sampler.set_epoch(epoch_x)
            data_iter_x = iter(data_loader_x)
            samples_x, targets_x = next(data_iter_x)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_u) + epoch, args)

        samples_x = samples_x.to(device, non_blocking=True)
        targets_x = targets_x.to(device, non_blocking=True)
        samples_u_w, samples_u_s = samples_u
        samples_u_w = samples_u_w.to(device, non_blocking=True)
        samples_u_s = samples_u_s.to(device, non_blocking=True)
        targets_u = targets_u.to(device, non_blocking=True)
        batch_size_x = samples_x.shape[0]

        if mixup_fn is not None and not args.disable_x_mixup:
            samples_x, targets_x = mixup_fn(samples_x, targets_x)

        with torch.cuda.amp.autocast():
            if args.drop > 0:
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.training = False
            if args.drop_path > 0 and args.disable_x_drop_path:
                for m in model.modules():
                    if isinstance(m, DropPath):
                        m.training = False
            logits_x = model(samples_x)
            if args.drop > 0:
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.training = True
            if args.drop_path > 0 and args.disable_x_drop_path:
                for m in model.modules():
                    if isinstance(m, DropPath):
                        m.training = True
            loss_x = criterion(logits_x, targets_x)
            # unlabeled data
            if epoch >= args.burnin_epochs:
                with torch.no_grad():
                    if args.ema_teacher:
                        logits_u_w = model_ema.ema(samples_u_w)
                    else:
                        logits_u_w = model(samples_u_w)
                # pseudo label
                pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
                max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                if pseudo_mixup_fn is not None:
                    if args.pseudo_mixup_func == "ProbPseudoMixup":
                        samples_u_s, pseudo_targets_u_mixup, max_probs = pseudo_mixup_fn(samples_u_s, pseudo_targets_u, max_probs)
                        mask = max_probs.ge(args.threshold).float()
                    else:
                        samples_u_s, pseudo_targets_u_mixup = pseudo_mixup_fn(samples_u_s, pseudo_targets_u)

                logits_u_s = model(samples_u_s)
                if pseudo_mixup_fn is not None:
                    loss_per_sample = torch.sum(-pseudo_targets_u_mixup * F.log_softmax(logits_u_s, dim=-1), dim=-1)
                else:
                    loss_per_sample = F.cross_entropy(logits_u_s, pseudo_targets_u, reduction='none')
                loss_u = (loss_per_sample * mask).mean()
            else:
                loss_u = 0.

            # overall losses
            loss = loss_x + args.lambda_u * loss_u

        loss_value = loss.item()
        loss_x_value = loss_x.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            print("loss_x: {}, loss_u: {}".format(loss_x, loss_u), force=True)
            print("======loss_per_sample_pseudo======", force=True)
            print(loss_per_sample, force=True)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc_x = (logits_x.max(-1)[-1] == targets_x).float().mean()
        else:
            class_acc_x = None

        loss_u_value, class_acc_u, mask_value = 0., 0., 0.
        pseudo_acc, pseudo_recall = 0., 0.
        if epoch >= args.burnin_epochs:
            loss_u_value = loss_u.item()
            pseudo_acc_batch = (pseudo_targets_u == targets_u).float()
            class_acc_u = pseudo_acc_batch.mean()
            if mask.sum() > 0:
                pseudo_acc = (pseudo_acc_batch * mask).sum() / mask.sum()
            if pseudo_acc_batch.sum() > 0:
                pseudo_recall = (pseudo_acc_batch * mask).sum() / pseudo_acc_batch.sum()
            mask_value = mask.mean().item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_x=loss_x_value)
        metric_logger.update(loss_u=loss_u_value)
        metric_logger.update(class_acc_x=class_acc_x)
        metric_logger.update(class_acc_u=class_acc_u)
        metric_logger.update(pseudo_acc=pseudo_acc)
        metric_logger.update(pseudo_recall=pseudo_recall)
        metric_logger.update(mask=mask_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader_u) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 100, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
