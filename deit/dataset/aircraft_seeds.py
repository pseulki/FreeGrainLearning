"""FGVC-Aircraft with the full 3-level hierarchy, generating superpixels (used for evaluation)."""
from typing import Optional, Callable, Any, Tuple, List, Union
from pathlib import Path
import os

import numpy as np
import torch
from torchvision.datasets import VisionDataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2
import PIL.Image

from data.air_get_tree_target import *


class FGVCAircraft(VisionDataset):
    def __init__(self,
                 root: Union[str, Path],
                 is_train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 mean: Union[List, Tuple] = IMAGENET_DEFAULT_MEAN,
                 std: Union[List, Tuple] = IMAGENET_DEFAULT_STD,
                 n_segments: int = 256,
                 compactness: float = 10.0,
                 blur_ops: Optional[Callable] = None,
                 scale_factor=1.0):
        super(FGVCAircraft, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform)

        self.mean = mean
        self.std = std
        self.n_segments = n_segments
        self.compactness = compactness
        self.blur_ops = blur_ops
        self.scale_factor = scale_factor

        self._data_path = os.path.join(self.root, "fgvc-aircraft-2013b")
        if not self._check_exists():
            raise RuntimeError("Dataset not found. Please download FGVC-Aircraft first.")

        # 'variant name' -> fine-grained id (1-indexed)
        with open('data/air-3L-variants.csv', 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        clsname_to_id = {}
        for line in lines:
            variant_name, variant_id = line.strip().split(",", 1)
            clsname_to_id[variant_name.strip('"')] = variant_id

        image_data_folder = os.path.join(self._data_path, "data", "images")
        if is_train:
            labels_file = os.path.join(self._data_path, "data", "images_variant_trainval.txt")
        else:
            labels_file = os.path.join(self._data_path, "data", "images_variant_test.txt")

        self._image_files = []
        self._labels = []
        with open(labels_file, "r") as f:
            for line in f:
                image_name, label_name = line.strip().split(" ", 1)
                self._image_files.append(os.path.join(image_data_folder, f"{image_name}.jpg"))
                self._labels.append(int(clsname_to_id[label_name]) - 1)

    def __len__(self) -> int:
        return len(self._labels)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, segments, fine-grained, subordinate, basic) targets.
        """
        path = self._image_files[index]
        target = self._labels[index]
        subord_target = trees[target][1] - 1
        basic_target = trees[target][2] - 1

        sample = PIL.Image.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

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

        return sample, segments, target, subord_target, basic_target
