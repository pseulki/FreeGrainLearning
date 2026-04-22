# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from mixup_hier import Mixup # we do not use mixup here. 
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

import torch.nn.functional as F
import time
from collections import Counter


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()


    for data in metric_logger.log_every(data_loader, print_freq, header):
        #if args.texts is not None:
        samples, targets, fine_targets, sub_targets, basic_targets, caps_embed = data
        caps_embed = caps_embed.to(device, non_blocking=True)
        # else:
        #     samples, targets, fine_targets, sub_targets, basic_targets = data
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        fine_targets = fine_targets.to(device, non_blocking=True)
        sub_targets = sub_targets.to(device, non_blocking=True)
        basic_targets = basic_targets.to(device, non_blocking=True)
        

        if 'BIRD' in args.data_set:
            leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
            sub_labels = torch.nonzero(targets > 12, as_tuple=False)

        elif 'IMNET-F' in args.data_set:
            leaf_labels = torch.nonzero(targets > 146, as_tuple=False)
            sub_labels = torch.nonzero(targets > 19, as_tuple=False)    

        elif 'AIR' in args.data_set:
            leaf_labels = torch.nonzero(targets > 99, as_tuple=False)
            sub_labels = torch.nonzero(targets > 29, as_tuple=False)    
        elif 'INAT21' in args.data_set:
                leaf_labels = torch.nonzero(targets > 1375, as_tuple=False)
                sub_labels = torch.nonzero(targets > 272, as_tuple=False)     
        
        else:
            raise ValueError('Unknown dataset')

        with torch.cuda.amp.autocast():
            out = model(samples)
            sim_loss = torch.tensor(0.0)  
            outputs, sub_out, basic_out, feats  = out

            feats = feats / feats.norm(dim=-1, keepdim=True)
            caps_embed = caps_embed / caps_embed.norm(dim=-1, keepdim=True) 
            labels = torch.arange(len(targets)).to(device)
            logits = torch.matmul(feats, caps_embed.t()) 
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.t(), labels)

            # Text-attr loss (CLIP style)
            sim_loss = (loss_i + loss_t) / 2
   
            
            loss_fine = 0  
            loss_sub = 0 
            loss_basic = 0

            if leaf_labels.shape[0] > 0:
                # supervision for samples who have fine-grained labels. 
                select_leaf_output = torch.index_select(outputs, 0, leaf_labels.squeeze())
                select_leaf_labels = torch.index_select(fine_targets, 0, leaf_labels.squeeze())
                loss_fine += (F.cross_entropy(select_leaf_output, select_leaf_labels))
    
            if sub_labels.shape[0] > 0:
                # supervision for samples who have subordinate labels. 
                select_sub_labels = torch.index_select(sub_targets, 0, sub_labels.squeeze())
                select_sub_output = torch.index_select(sub_out, 0, sub_labels.squeeze())
                loss_sub += (F.cross_entropy(select_sub_output, select_sub_labels))

            loss_basic = (F.cross_entropy(basic_out, basic_targets))

                
        loss = loss_fine + loss_sub + loss_basic + sim_loss * args.sim_loss_weight 
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(sp_loss=loss_fine.item())
        metric_logger.update(subord_loss=loss_sub.item())
        metric_logger.update(basic_loss=loss_basic.item())
        metric_logger.update(sim_loss=sim_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        del feats, caps_embed, loss_i, loss_t, sim_loss, logits
        del samples, targets, outputs, loss
        torch.cuda.empty_cache()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
   

@torch.no_grad()
def evaluate(data_loader, model, device, n_classes=3, texts=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target, sub_targets, basic_targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sub_targets = sub_targets.to(device, non_blocking=True)
        basic_targets = basic_targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, sub_out, basic_out, *rest  = model(images)

            loss_fine = criterion(output, target)
            loss_sub = criterion(sub_out, sub_targets)
            loss_basic = criterion(basic_out, basic_targets)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        sub_acc1, sub_acc5 = accuracy(sub_out, sub_targets, topk=(1, 5))
        basic_acc1, basic_acc5 = accuracy(basic_out, basic_targets, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(sploss=loss_fine.item())
        metric_logger.update(subordloss=loss_sub.item())
        metric_logger.update(manuloss=loss_basic.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['sub_acc1'].update(sub_acc1.item(), n=batch_size)
        metric_logger.meters['basic_acc1'].update(basic_acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} sub@1 {subtop1.global_avg:.3f}' 
        ' manu@1 {manutop1.global_avg:.3f} sploss {losses.global_avg:.3f} fmloss {fmlosses.global_avg:.3f} basicloss {basiclosses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.sploss, fmlosses=metric_logger.subordloss, basiclosses=metric_logger.manuloss,
                subtop1=metric_logger.sub_acc1, manutop1=metric_logger.basic_acc1))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

