"""iNat21-Mini with the full 3-level hierarchy (used for evaluation)."""
from typing import Optional, Callable, Any, Tuple, List, Union
import os

from torch.utils.data import Dataset
from PIL import Image


class iNat21MiniDataset(Dataset):
    def __init__(self,
                 root,
                 is_train: bool = True,
                 transform=None,):
        self.transform = transform

        self.img_path = []
        self.basic_label_list = []
        self.subord_label_list = []
        self.class_label_list = []

        if is_train:
            txt = os.path.join('data/inat21-F-train.txt')
        else:
            txt = os.path.join('data/inat21-F-val.txt')

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.basic_label_list.append(int(line.split()[1]))
                self.subord_label_list.append(int(line.split()[2]))
                self.class_label_list.append(int(line.split()[3]))

    def __len__(self):
        return len(self.class_label_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, fine-grained, subordinate, basic) targets.
        """
        path = self.img_path[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.class_label_list[index], self.subord_label_list[index], self.basic_label_list[index]
