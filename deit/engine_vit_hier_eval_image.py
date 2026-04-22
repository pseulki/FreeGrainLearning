# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import json
import csv
import numpy as np


@torch.no_grad()
def evaluate_detail(data_loader, model, device, filename, n_classes=3, dataset='AIR-HIER', texts=None, out_embedding=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if 'INAT21' in dataset:
        inat_trees = json.load(open('data/inat21-F-tree.json')) 

    elif 'IMNET-F' in dataset:
        imnet_f_trees = json.load(open('data/imagenet-3L-tree.json'))



    # switch to evaluation mode
    model.eval()
    results = []
    tice_cnt = 0
    fpa_cnt = 0
    total_cnt = 0

    results.append(['basic_gt', 'basic_pred', 'sub_gt', 'sub_pred', 'fine_gt', 'fine_pred'])
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
        metric_logger.update(basicloss=loss_basic.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['sub_acc1'].update(sub_acc1.item(), n=batch_size)
        metric_logger.meters['basic_acc1'].update(basic_acc1.item(), n=batch_size)

        _, pred = torch.max(output, 1)
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        _, sub_pred = torch.max(sub_out, 1)
        sub_pred = sub_pred.cpu().numpy()
        sub_targets = sub_targets.cpu().numpy()

        _, basic_pred = torch.max(basic_out, 1)
        basic_pred = basic_pred.cpu().numpy()
        basic_targets = basic_targets.cpu().numpy()


        total_cnt += batch_size
        for i in range(batch_size):
            results.append([basic_targets[i], basic_pred[i], sub_targets[i], sub_pred[i], target[i], pred[i]])
            tice_results = [pred[i]+1, sub_pred[i]+1, basic_pred[i]+1]
            
            if pred[i] == target[i] and sub_pred[i] == sub_targets[i] and basic_pred[i] == basic_targets[i]:
                fpa_cnt += 1

            if 'AIR' in dataset:
                tice_results = [pred[i]+1, sub_pred[i]+1, basic_pred[i]+1]
                if tice_results in air_trees:
                    tice_cnt += 1
            elif 'BIRD' in dataset:
                tice_results = [pred[i]+1, basic_pred[i]+1, sub_pred[i]+1]
                if tice_results in birds_trees:
                    tice_cnt += 1
            elif 'INAT21' in dataset:
                tice_results = [pred[i], sub_pred[i], basic_pred[i]]
                if tice_results in inat_trees:
                    tice_cnt += 1
            elif 'IMNET-F' in dataset:
                tice_results = [pred[i], sub_pred[i], basic_pred[i]]
                if tice_results in imnet_f_trees:
                    tice_cnt += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} sub@1 {subtop1.global_avg:.3f}' 
        ' basic@1 {basictop1.global_avg:.3f} sploss {losses.global_avg:.3f} subordloss {subordlosses.global_avg:.3f} basicloss {basiclosses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.sploss, subordlosses=metric_logger.subordloss, basiclosses=metric_logger.basicloss,
                subtop1=metric_logger.sub_acc1, basictop1=metric_logger.basic_acc1))
    
    print(f"FPA: {(fpa_cnt / total_cnt) * 100:.3f}% | TICE: {((total_cnt - tice_cnt) / total_cnt) * 100:.3f}% ")

    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerows(results)
        csvwriter.writerows(str(tice_cnt))
    
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

air_trees = [
[1, 1, 1],
[2, 2, 1],
[3, 3, 1],
[4, 3, 1],
[5, 3, 1],
[6, 3, 1],
[7, 4, 1],
[8, 4, 1],
[9, 5, 1],
[10, 5, 1],
[11, 5, 1],
[12, 5, 1],
[13, 6, 1],
[14, 7, 2],
[15, 8, 3],
[16, 9, 3],
[17, 10, 7],
[18, 10, 7],
[19, 11, 7],
[20, 12, 4],
[21, 13, 5],
[22, 14, 5],
[23, 15, 5],
[24, 16, 5],
[25, 16, 5],
[26, 16, 5],
[27, 16, 5],
[28, 16, 5],
[29, 16, 5],
[30, 16, 5],
[31, 16, 5],
[32, 17, 5],
[33, 17, 5],
[34, 17, 5],
[35, 17, 5],
[36, 18, 5],
[37, 18, 5],
[38, 19, 5],
[39, 19, 5],
[40, 19, 5],
[41, 20, 5],
[42, 20, 5],
[43, 21, 21],
[44, 22, 14],
[45, 23, 9],
[46, 24, 9],
[47, 25, 9],
[48, 25, 9],
[49, 26, 8],
[50, 27, 8],
[51, 28, 8],
[52, 28, 8],
[53, 29, 12],
[54, 29, 12],
[55, 30, 23],
[56, 31, 14],
[57, 32, 14],
[58, 33, 14],
[59, 34, 23],
[60, 35, 12],
[61, 36, 12],
[62, 37, 12],
[63, 38, 13],
[64, 39, 26],
[65, 40, 15],
[66, 41, 15],
[67, 41, 15],
[68, 41, 15],
[69, 42, 15],
[70, 42, 15],
[71, 43, 15],
[72, 44, 16],
[73, 45, 23],
[74, 46, 22],
[75, 47, 11],
[76, 48, 11],
[77, 49, 18],
[78, 50, 18],
[79, 51, 18],
[80, 52, 6],
[81, 53, 19],
[82, 53, 19],
[83, 54, 7],
[84, 55, 20],
[85, 56, 4],
[86, 57, 21],
[87, 58, 23],
[88, 59, 23],
[89, 59, 23],
[90, 60, 23],
[91, 61, 17],
[92, 62, 25],
[93, 63, 27],
[94, 64, 27],
[95, 65, 28],
[96, 66, 10],
[97, 67, 24],
[98, 68, 29],
[99, 69, 29],
[100, 70, 30]
]


birds_trees = [
[1,12,35],
[2,12,35],
[3,12,35],
[4,6,9],
[5,4,4],
[6,4,4],
[7,4,4],
[8,4,4],
[9,8,18],
[10,8,18],
[11,8,18],
[12,8,18],
[13,8,18],
[14,8,13],
[15,8,13],
[16,8,13],
[17,8,13],
[18,8,26],
[19,8,21],
[20,8,19],
[21,8,24],
[22,3,3],
[23,13,37],
[24,13,37],
[25,13,37],
[26,8,18],
[27,8,18],
[28,8,14],
[29,8,15],
[30,8,15],
[31,6,9],
[32,6,9],
[33,6,9],
[34,8,16],
[35,8,16],
[36,10,33],
[37,8,30],
[38,8,30],
[39,8,30],
[40,8,30],
[41,8,30],
[42,8,30],
[43,8,30],
[44,13,38],
[45,12,36],
[46,1,1],
[47,8,16],
[48,8,16],
[49,8,18],
[50,11,34],
[51,11,34],
[52,11,34],
[53,11,34],
[54,8,13],
[55,8,16],
[56,8,16],
[57,8,13],
[58,4,4],
[59,4,5],
[60,4,5],
[61,4,5],
[62,4,5],
[63,4,5],
[64,4,5],
[65,4,5],
[66,4,5],
[67,2,2],
[68,2,2],
[69,2,2],
[70,2,2],
[71,4,6],
[72,4,6],
[73,8,15],
[74,8,15],
[75,8,15],
[76,8,24],
[77,8,30],
[78,8,30],
[79,5,7],
[80,5,7],
[81,5,7],
[82,5,7],
[83,5,7],
[84,5,8],
[85,8,11],
[86,7,10],
[87,1,1],
[88,8,18],
[89,1,1],
[90,1,1],
[91,8,21],
[92,3,3],
[93,8,15],
[94,8,27],
[95,8,18],
[96,8,18],
[97,8,18],
[98,8,18],
[99,8,23],
[100,9,32],
[101,9,32],
[102,8,30],
[103,8,30],
[104,8,22],
[105,3,3],
[106,4,4],
[107,8,15],
[108,8,15],
[109,8,23],
[110,6,9],
[111,8,20],
[112,8,20],
[113,8,24],
[114,8,24],
[115,8,24],
[116,8,24],
[117,8,24],
[118,8,25],
[119,8,24],
[120,8,24],
[121,8,24],
[122,8,24],
[123,8,24],
[124,8,24],
[125,8,24],
[126,8,24],
[127,8,24],
[128,8,24],
[129,8,24],
[130,8,24],
[131,8,24],
[132,8,24],
[133,8,24],
[134,8,28],
[135,8,17],
[136,8,17],
[137,8,17],
[138,8,17],
[139,8,13],
[140,8,13],
[141,4,5],
[142,4,5],
[143,4,5],
[144,4,5],
[145,4,5],
[146,4,5],
[147,4,5],
[148,8,24],
[149,8,21],
[150,8,21],
[151,8,31],
[152,8,31],
[153,8,31],
[154,8,31],
[155,8,31],
[156,8,31],
[157,8,31],
[158,8,23],
[159,8,23],
[160,8,23],
[161,8,23],
[162,8,23],
[163,8,23],
[164,8,23],
[165,8,23],
[166,8,23],
[167,8,23],
[168,8,23],
[169,8,23],
[170,8,23],
[171,8,23],
[172,8,23],
[173,8,23],
[174,8,23],
[175,8,23],
[176,8,23],
[177,8,23],
[178,8,23],
[179,8,23],
[180,8,23],
[181,8,23],
[182,8,23],
[183,8,23],
[184,8,23],
[185,8,12],
[186,8,12],
[187,10,33],
[188,10,33],
[189,10,33],
[190,10,33],
[191,10,33],
[192,10,33],
[193,8,29],
[194,8,29],
[195,8,29],
[196,8,29],
[197,8,29],
[198,8,29],
[199,8,29],
[200,8,23]
]