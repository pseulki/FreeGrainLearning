from typing import Optional, Callable, Any, Tuple, List, Union
import os
import torch
from torch.utils.data import DataLoader, Dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import numpy as np
import random
import json
import cv2

class ImageNetHier(Dataset):
    def __init__(self, 
                 root, 
                 is_train: bool = True,
                 transform=None,
                 path_yn: bool = False,
                 txt: str = 'imagenet-HC.txt',
                 nb_classes=[505, 127, 20],
                 mean: Union[List, Tuple] = IMAGENET_DEFAULT_MEAN,
                 std: Union[List, Tuple] = IMAGENET_DEFAULT_STD,
                 n_segments: int = 256,
                 compactness: float = 10.0,
                 blur_ops: Optional[Callable] = None,
                 scale_factor=1.0,):

        self.mean = mean
        self.std = std
        self.n_segments = n_segments
        self.compactness = compactness
        self.blur_ops = blur_ops
        self.scale_factor = scale_factor


        self.transform = transform
        self.img_path = []
        self.path_yn = path_yn
        self.basic_label_list = []
        self.subord_label_list = []
        self.class_label_list = []
        self.labels = []
        self.is_train = is_train

        if is_train:
            txt = os.path.join('data/imagenet-F-train.txt')
            with open(txt) as f:
                for line in f:
                    self.img_path.append(os.path.join(root, line.split()[0]))
                    self.basic_label_list.append(int(line.split()[1]))
                    self.subord_label_list.append(int(line.split()[2]))
                    self.class_label_list.append(int(line.split()[3]))
                    self.labels.append(int(line.split()[4]))  
        else:
            txt = os.path.join('data/imagenet-F-val.txt')
            with open(txt) as f:
                for line in f:
                    self.img_path.append(os.path.join(root, line.split()[0])) 
                    self.basic_label_list.append(int(line.split()[1]))
                    self.subord_label_list.append(int(line.split()[2]))
                    self.class_label_list.append(int(line.split()[3]))

    def __len__(self):
        return len(self.class_label_list)


    def __getitem__(self, index):

        path = self.img_path[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        
        # Prepare arguments when multi-view pipeline is adopted.
        compactness = self.compactness
        blur_ops = self.blur_ops
        n_segments = self.n_segments
        scale_factor = self.scale_factor
        if isinstance(sample, (list, tuple)):
            if not isinstance(compactness, (list, tuple)):
                compactness = [compactness] * len(sample)

            if not isinstance(n_segments, (list, tuple)):
                n_segments = [n_segments] * len(sample)

            if not isinstance(blur_ops, (list, tuple)):
                blur_ops = [blur_ops] * len(sample)

            if not isinstance(scale_factor, (list, tuple)):
                scale_factor = [scale_factor] * len(sample)


        # Generate superpixels.
        if isinstance(sample, (list, tuple)):
            segments = []
            for samp, comp, n_seg, blur_op, scale in zip(sample, compactness, n_segments, blur_ops, scale_factor):
                if blur_op is not None:
                    samp = blur_op(samp)
                samp = (samp.data.numpy().transpose(1, 2, 0) * self.std + self.mean)
                samp = (samp * 255).astype(np.uint8)
                samp = cv2.cvtColor(samp, cv2.COLOR_RGB2LAB)
                seeds = cv2.ximgproc.createSuperpixelSEEDS(
                    samp.shape[1], samp.shape[0], 3, num_superpixels=self.n_segments, num_levels=1, prior=2,
                    histogram_bins=5, double_step=False);
                seeds.iterate(samp, num_iterations=15);
                segment = seeds.getLabels()
                segment = torch.LongTensor(segment)
                segments.append(segment)
        else:
            if blur_ops is not None:
                samp = blur_ops(sample)
            else:
              samp = sample
            samp = (samp.data.numpy().transpose(1, 2, 0) * self.std + self.mean)
            samp = (samp * 255).astype(np.uint8)
            samp = cv2.cvtColor(samp, cv2.COLOR_RGB2LAB)
            seeds = cv2.ximgproc.createSuperpixelSEEDS(
                samp.shape[1], samp.shape[0], 3, num_superpixels=self.n_segments, num_levels=1, prior=2,
                histogram_bins=5, double_step=False);
            seeds.iterate(samp, num_iterations=15);
            segments = seeds.getLabels()
            segments = torch.LongTensor(segments)


        
        if self.is_train:
            return sample, segments, self.labels[index], self.class_label_list[index], self.subord_label_list[index], self.basic_label_list[index]

        else:
            return sample, segments, self.class_label_list[index], self.subord_label_list[index], self.basic_label_list[index]
