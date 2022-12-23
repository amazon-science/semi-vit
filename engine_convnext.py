# ------------------------------------------------------------------------
# Modified from ConvNeXt (https://github.com/facebookresearch/ConvNeXt)
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Iterable, Optional
import wandb

import torch
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import util.misc as utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else:  # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):  # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:  # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_ssl(
        model: torch.nn.Module, criterion: torch.nn.Module, data_loader_x: Iterable, data_loader_u: Iterable,
        optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, pseudo_mixup_fn=None, log_writer=None,
        wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
        num_training_steps_per_epoch=None, update_freq=None, use_amp=False, args=None
):
    model.train(True)
    model_ema.ema.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    epoch_x = epoch * math.ceil(len(data_loader_u) / len(data_loader_x))
    if args.distributed:
        print("set epoch={} for labeled sampler".format(epoch_x))
        data_loader_x.sampler.set_epoch(epoch_x)
        print("set epoch={} for unlabeled sampler".format(epoch))
        data_loader_u.sampler.set_epoch(epoch)

    optimizer.zero_grad()

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

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

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
            logits_x = model(samples_x)
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

        if not math.isfinite(loss_value):  # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value), force=True)
            print("loss_x: {}, loss_u: {}".format(loss_x, loss_u), force=True)
            print("======loss_per_sample_pseudo======", force=True)
            print(loss_per_sample, force=True)
            assert math.isfinite(loss_value)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
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
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc_x=class_acc_x, head="loss")
            log_writer.update(class_acc_u=class_acc_u, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if data_iter_step % args.print_freq == 0 and args.wandb:
            wandb_log_dict = {
                'loss': metric_logger.meters['loss'].median, 'loss_x': metric_logger.meters['loss_x'].median,
                'loss_u': metric_logger.meters['loss_u'].median,
                'class_acc_u': metric_logger.meters['class_acc_u'].median,
                'mask': metric_logger.meters['mask'].median, 'pseudo_acc': metric_logger.meters['pseudo_acc'].median,
                'pseudo_recall': metric_logger.meters['pseudo_recall'].median,
            }
            if class_acc_x is not None:
                wandb_log_dict.update({'class_acc_x': metric_logger.meters['class_acc_x'].median})
            wandb.log(wandb_log_dict, commit=True)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 100, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
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
