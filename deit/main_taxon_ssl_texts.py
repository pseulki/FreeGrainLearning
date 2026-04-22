# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# CHMatch module from: https://github.com/sailist/image-classification (We appreciate the authors for codes!)


import os
import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
import json

from pathlib import Path

from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from lumo.contrib.nn.loss import contrastive_loss2

from datasets_partial import build_dataset
from datasets import build_dataset as build_dataset_test

from engine_vit_hier_partial import evaluate
from engine_vit_hier_eval_image import evaluate_detail
from samplers import RASampler


import utils

from randaugment import RandAugment


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['IMNET-F'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--breeds_sort', default='entity13', type=str, choices=['entity13', 'living17', 'nonliving26', 'entity30'])
    parser.add_argument('--issource', action='store_false')
    parser.add_argument('--path-yn', action='store_true')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--filename', default='reverse_best.csv', type=str)
    parser.add_argument('--imb_type', default='exp', type=str, choices=['exp', 'bal'])
    parser.add_argument('--img_max', default=None, type=int)
    parser.add_argument('--sourcefile', default='_train_source.txt', type=str)
    parser.add_argument('--random_seed', default=1, type=int)
    parser.add_argument('--sim_loss_weight', default=1.0, type=float)
    parser.add_argument('--texts', default=None, type=str)

    parser.add_argument('--sp_proportion', default=0.25, type=float)
    parser.add_argument('--fm_proportion', default=0.25, type=float)
    parser.add_argument('--rand_number', default=0, type=int)
    parser.add_argument('--re_level', default='family', type=str, choices=['family', 'order', 'species'])

    # text loss
    parser.add_argument('--text_loss_weight', default=0.0, type=float)
    parser.add_argument('--HierViT', action='store_true')
    
    return parser


class MLP(nn.Module):
    """adapted from https://github.com/p3i0t/SimCLR-CIFAR10/blob/master/models.py"""

    def __init__(self, feature_dim, mid_dim, output_dim, with_bn=False, with_leakyrelu=True):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(feature_dim, mid_dim),
            nn.BatchNorm1d(mid_dim) if with_bn else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) if with_leakyrelu else nn.ReLU(inplace=True),
            nn.Linear(mid_dim, output_dim),
        )

    def forward(self, feature):
        return self.module(feature)


class CHMatchModule(nn.Module):

    def __init__(self, model, nb_classes, is_hier):
        super(CHMatchModule, self).__init__()
        self.model = model
        self.nb_classes = nb_classes
        self.is_hier = is_hier

        if not is_hier:
            feature_dim = self.model.blocks[-1].attn.qkv.in_features
            self.mlp = MLP(feature_dim, feature_dim, feature_dim, with_bn=True)
            self.classifiers = nn.ModuleList([nn.Linear(feature_dim, nb_class) for nb_class in self.nb_classes])
            self.text_head = nn.Linear(feature_dim, 512)

        # data alignment
        self.data_align = [[] for _ in range(len(self.nb_classes))]

        # memory bank
        self.memory_bank = [[] for _ in range(len(self.nb_classes))]

    def forward(self, x):
        if self.is_hier:
            return self.model(x)
        else:
            feats = self.model(x)
            x = self.mlp(feats)
            logits = [classifier(x) for classifier in self.classifiers]
            text_feats = self.text_head(feats)
            return (*logits, feats, text_feats) # including the featus as well to match the HierVisionTransformer
    
    def update_memory_bank(self, scores):
        for i, score in enumerate(scores):
            self.memory_bank[i].extend(score.cpu().tolist())
            if len(self.memory_bank[i]) > 50000:
                self.memory_bank[i] = self.memory_bank[i][-50000:]


class KScheduler():
    def __init__(self, init, end, max_epoch):
        self.init = init
        self.end = end
        self.max_epoch = max_epoch
        self.slope = (end - init) / max_epoch
    def __call__(self, epoch):
        return min(self.end, self.init + self.slope * epoch)
    

def sharpen(x: torch.Tensor, T=0.5):
    """
    Sharpen the distribution to be closer to one-hot

    Inputs:
        - x (torch.Tensor): prediction, sum(x,dim=-1) = 1
        - T (optional, float): temperature, default is 0.5
    """
    with torch.no_grad():
        temp = torch.pow(x, 1 / T)
        return temp / (temp.sum(dim=1, keepdims=True) + 1e-7)



def train_one_epoch(model, dataloader, k_value, optimizer, device, epoch, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Training epoch {}:'.format(epoch)
    model.train()
    model.to(device)

    for images, labels, species_labels, family_labels, order_labels, caps_embed in metric_logger.log_every(dataloader, 10, header):
        standard_images = images['standard'].to(device, non_blocking=True)
        simclr_images = images['simclr'].to(device, non_blocking=True)
        randaug_images = images['randaug'].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        species_labels = species_labels.to(device, non_blocking=True)
        family_labels = family_labels.to(device, non_blocking=True)
        order_labels = order_labels.to(device, non_blocking=True)
        caps_embed = caps_embed.to(device, non_blocking=True)

        # forward pass
        output = model(torch.cat([standard_images, simclr_images, randaug_images], dim=0))
        species_logits, family_logits, order_logits, feats, text_feats = output
        standard_text_feats, _, _ = text_feats.chunk(3)
        standard_species_logits, simclr_species_logits, _ = species_logits.chunk(3)
        standard_family_logits, simclr_family_logits, _ = family_logits.chunk(3)
        standard_order_logits, simclr_order_logits, _ = order_logits.chunk(3)
        _, simclr_feats, randaug_feats = feats.chunk(3)

        # text loss
        loss_text = torch.tensor(0.0).to(device)
        if standard_text_feats is not None and args.text_loss_weight > 0:
            with torch.cuda.amp.autocast():
                standard_text_feats = standard_text_feats / standard_text_feats.norm(dim=-1, keepdim=True)
                caps_embed = caps_embed / caps_embed.norm(dim=-1, keepdim=True) 
                text_labels = torch.arange(len(labels)).to(device)
                logits = torch.matmul(standard_text_feats, caps_embed.t()) 
                loss_i = nn.CrossEntropyLoss()(logits, text_labels)
                loss_t = nn.CrossEntropyLoss()(logits.t(), text_labels)
                loss_text = (loss_i + loss_t) / 2

        # supervised loss
        if 'BIRD' in args.data_set:
            species_indices = torch.nonzero(labels > 50, as_tuple=False).squeeze()
            family_indices = torch.nonzero(labels > 12, as_tuple=False).squeeze()
        elif 'IMNET-F' in args.data_set:
            species_indices = torch.nonzero(labels > 146, as_tuple=False).squeeze()
            family_indices = torch.nonzero(labels > 19, as_tuple=False).squeeze()
        elif 'AIR' in args.data_set:
            species_indices = torch.nonzero(labels > 99, as_tuple=False).squeeze()
            family_indices = torch.nonzero(labels > 29, as_tuple=False).squeeze()
        else:
            raise NotImplementedError('Dataset not supported')
        # when there's exactly one label in the batch, the dimension of the tensor will be 0, e.g., torch.tensor(0)
        if species_indices.dim() == 0:
            species_indices = species_indices.unsqueeze(0)
        if family_indices.dim() == 0:
            family_indices = family_indices.unsqueeze(0)

        loss_supervised = 0.0
        if len(species_indices) > 0:
            loss_supervised += nn.CrossEntropyLoss()(standard_species_logits[species_indices], species_labels[species_indices])
        if len(family_indices) > 0:
            loss_supervised += nn.CrossEntropyLoss()(standard_family_logits[family_indices], family_labels[family_indices])
        loss_supervised += nn.CrossEntropyLoss()(standard_order_logits, order_labels)

        # psuedo-labeling loss
        with torch.no_grad():
            # obtain class probabilities
            standard_species_probs = torch.softmax(standard_species_logits, dim=1)
            standard_family_probs = torch.softmax(standard_family_logits, dim=1)
            standard_order_probs = torch.softmax(standard_order_logits, dim=1)
            
            # data alignment
            new_probs = []
            for i, probs in enumerate([standard_species_probs, standard_family_probs, standard_order_probs]):
                model.data_align[i].append(probs.mean(0))
                if len(model.data_align[i]) > 32:
                    model.data_align[i].pop(0)
                prob_avg = torch.stack(model.data_align[i]).mean(0)
                probs = probs / prob_avg
                probs = probs / probs.sum(dim=1, keepdim=True)
                new_probs.append(probs)

            # score and psuedo-labeling
            score_pl = [torch.max(it, dim=1) for it in new_probs]
            scores = [it.values for it in score_pl]
            pseudo_labels = [it.indices for it in score_pl]
            model.update_memory_bank(scores)
            threshes = [torch.tensor(it).topk(int(len(it) * k_value)).values[-1] for it in model.memory_bank]
            p_masks = [score.ge(thresh) for score, thresh in zip(scores, threshes)]
        
        simclr_logits = [simclr_species_logits, simclr_family_logits, simclr_order_logits]
        loss_pl = torch.zeros_like(pseudo_labels[0]).float()
        for logit, pseudo_label, p_mask in zip(simclr_logits, pseudo_labels, p_masks):
            loss_pl += torch.nn.functional.cross_entropy(logit, pseudo_label, reduction='none') * p_mask.float()
        loss_pl = loss_pl.mean()


        # contrastive loss
        with torch.no_grad():
            graphs = [torch.eq(pseudo_label[:, None], pseudo_label[None, :]) for pseudo_label in pseudo_labels]
            graph = torch.prod(torch.stack(graphs), dim=0).bool()
            graph = graph | torch.eye(len(graph), device=device, dtype=torch.bool)
            graph = sharpen(graph, 1)
        loss_contrastive = contrastive_loss2(simclr_feats, randaug_feats, qk_graph=graph, norm=True, temperature=0.1)

        # overall loss
        loss_all = loss_supervised + loss_pl + loss_contrastive + loss_text * args.text_loss_weight

        # backward pass
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        # log
        metric_logger.update(loss_text=loss_text.item())
        metric_logger.update(loss_supervised=loss_supervised.item())
        metric_logger.update(loss_pl=loss_pl.item())
        metric_logger.update(loss_contrastive=loss_contrastive.item())
        metric_logger.update(loss_all=loss_all.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(k=k_value)
        metric_logger.update(specifies_thresh=threshes[0].item())
        metric_logger.update(families_thresh=threshes[1].item())
        metric_logger.update(orders_thresh=threshes[2].item())
        with torch.no_grad():
            species_acc = (torch.argmax(standard_species_logits[species_indices], dim=1) == species_labels[species_indices]).float().mean().item()
            family_acc = (torch.argmax(standard_family_logits[family_indices], dim=1) == family_labels[family_indices]).float().mean().item()
            order_acc = (torch.argmax(standard_order_logits, dim=1) == order_labels).float().mean().item()
            metric_logger.update(species_acc=species_acc)
            metric_logger.update(family_acc=family_acc)
            metric_logger.update(order_acc=order_acc)

            species_pl_acc = (pseudo_labels[0] == species_labels).float().mean().item()
            family_pl_acc = (pseudo_labels[1] == family_labels).float().mean().item()
            order_pl_acc = (pseudo_labels[2] == order_labels).float().mean().item()
            metric_logger.update(species_pl_acc=species_pl_acc)
            metric_logger.update(family_pl_acc=family_pl_acc)
            metric_logger.update(order_pl_acc=order_pl_acc)

            float_graphs = [graph.float() for graph in graphs]
            species_family_discrepency = ((torch.abs(float_graphs[1]-float_graphs[0])>0).sum())/(float_graphs[0].sum()+float_graphs[1].sum() - 2*len(float_graphs[0]))
            family_order_discrepency = ((torch.abs(float_graphs[2]-float_graphs[1])>0).sum())/(float_graphs[1].sum()+float_graphs[2].sum() - 2*len(float_graphs[1]))
            metric_logger.update(species_family_discrepency=species_family_discrepency.item())
            metric_logger.update(family_order_discrepency=family_order_discrepency.item())

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    print(args)
    if args.distributed:
        utils.init_distributed_mode(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    if args.distributed:
        seed = args.seed + utils.get_rank()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset_test(is_train=False, args=args)

    nm = (2,10)
    chmatch_transform = {
        'standard': transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(args.input_size, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]),

        'simclr': transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.RandomResizedCrop(args.input_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]),

        'randaug': transforms.Compose([
            transforms.Resize(args.input_size),
            RandAugment(n=nm[0], m=nm[1]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(args.input_size, padding=args.input_size // 8, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]),

    }
    dataset_train.transform = chmatch_transform
    
    print('args.nb_classes', args.nb_classes)

    

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model: {args.model}")
    # if using HierViT
    if args.HierViT:
        import models_hier
        import models_v2
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes[0],
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size,
            nb_classes=args.nb_classes
        )
    else:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=0,
            # num_classes=args.nb_classes[0],
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size,
            # nb_classes=args.nb_classes
    )
                    
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')

    model = CHMatchModule(model, args.nb_classes, is_hier=args.HierViT)
            
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Training tools
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    k_scheduler = KScheduler(0.05, 0.8, args.epochs // 4)

    output_dir = Path(args.output_dir)
    if args.eval:
        test_stats = evaluate_detail(data_loader_val, model, device, os.path.join(args.output_dir, args.filename), len(args.nb_classes), args.data_set, args.breeds_sort)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        test_log_stats = {
            # **{f'train_{k}': v for k, v in train_stats.items()}, # to be uncommented
            **{f'test_{k}': v for k, v in test_stats.items()},
            'n_parameters': n_parameters
        }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "test_log_detail.txt").open("a") as f:
                f.write(json.dumps(test_log_stats) + "\n")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        k_value = k_scheduler(epoch)
        train_stats = train_one_epoch(model, data_loader_train, k_value, optimizer, device, epoch, args)
        lr_scheduler.step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'accuracy': max_accuracy,
                    'args': args,
                }, checkpoint_path)
             

        test_stats = evaluate(data_loader_val, model, device, len(args.nb_classes))
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'accuracy': max_accuracy,
                        'args': args,
                    }, checkpoint_path)
            
        print(f'Max accuracy: {max_accuracy:.2f}%')

        test_log_stats = {
            # **{f'train_{k}': v for k, v in train_stats.items()}, # to be uncommented
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "test_log.txt").open("a") as f:
                f.write(json.dumps(test_log_stats) + "\n")


        train_log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "train_log.txt").open("a") as f:
                f.write(json.dumps(train_log_stats) + "\n")

        if epoch % 10 == 0:
            test_stats = evaluate_detail(data_loader_val, model, device, os.path.join(args.output_dir, args.filename), len(args.nb_classes), args.data_set, args.breeds_sort)

            test_log_stats = {
                # **{f'train_{k}': v for k, v in train_stats.items()}, # to be uncommented
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }
            if args.output_dir and utils.is_main_process():
                with (output_dir / "test_log_detail.txt").open("a") as f:
                    f.write(json.dumps(test_log_stats) + "\n")
        
        print('\n----------\n')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    test_stats = evaluate_detail(data_loader_val, model, device, os.path.join(args.output_dir, args.filename), len(args.nb_classes), args.data_set, args.breeds_sort)
    test_log_stats = {
        # **{f'train_{k}': v for k, v in train_stats.items()}, # to be uncommented
        **{f'test_{k}': v for k, v in test_stats.items()},
        'epoch': epoch,
        'n_parameters': n_parameters
    }
    if args.output_dir and utils.is_main_process():
        with (output_dir / "test_log_detail.txt").open("a") as f:
            f.write(json.dumps(test_log_stats) + "\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
