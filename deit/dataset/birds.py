"""CUB-200-2011 with the full 3-level hierarchy (used for evaluation)."""
from typing import Optional, Callable, Any, Tuple, List, Union

import torchvision.datasets as datasets
import torchvision.datasets.folder as folder

from data.birds_get_tree_target_2 import *


class ImageFolder(datasets.ImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = folder.default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None):
        super(ImageFolder, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, fine-grained, subordinate, basic) targets.
        """
        path, target = self.samples[index]
        basic_target = trees[target][1] - 1
        subord_target = trees[target][2] - 1

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, subord_target, basic_target
