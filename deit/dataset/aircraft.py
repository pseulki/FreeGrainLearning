"""FGVC-Aircraft with the full 3-level hierarchy (used for evaluation)."""
from typing import Optional, Callable, Any, Tuple, List, Union
from pathlib import Path
import os

from torchvision.datasets import VisionDataset
import PIL.Image

from data.air_get_tree_target import *


class FGVCAircraft_Hier(VisionDataset):
    def __init__(self,
                 root: Union[str, Path],
                 is_train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super(FGVCAircraft_Hier, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform)

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
            tuple: (sample, fine-grained, subordinate, basic) targets.
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

        return sample, target, subord_target, basic_target
