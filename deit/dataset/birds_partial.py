"""CUB-200-2011 under the free-grain setting (CUB-F, synthesized).

The label granularity of each training image is drawn according to
`sp_proportion` / `fm_proportion`, so that a sample is supervised at the
fine-grained (species), subordinate (family) or basic (order) level.
The given label is encoded as
    0-12    -> basic level only
    13-50   -> up to subordinate level
    51-250  -> fine-grained level
"""
from typing import Optional, Callable, Any, Tuple, List, Union

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.datasets.folder as folder

import clip

from data.birds_get_tree_target_2 import *


class ImageFolder(datasets.ImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = folder.default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 is_train: bool = True,
                 random_number: int = 0,
                 sp_proportion=0.1,
                 fm_proportion=0.55,
                 texts: str = None,
                 clip_model: str = "ViT-B/32",):
        super(ImageFolder, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file)

        np.random.seed(random_number)
        self.sp_proportion = sp_proportion
        self.fm_proportion = fm_proportion
        self.texts = texts

        self.image_filenames, self.labels, self.fine_labels, self.subord_labels, self.basic_labels = self.relabel()

        if is_train:
            if texts:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print('device', device)
                model, _ = clip.load(clip_model, device)
                model.eval()

                #text
                cap_dic = {}
                f = open(texts, 'r')
                lines = f.readlines()
                f.close()
                for line in lines:
                    id = line.split('.jpg, ')[0].strip() + '.jpg'
                    cap = line.split('.jpg, ')[1].strip()
                    cap_dic[id] = cap

                self.caps = []
                for i in range(len(self.image_filenames)):
                    id = "/".join(self.image_filenames[i].split('/')[-2:])
                    self.caps.append(cap_dic[id])

                self.cap_embs = []
                num_text = len(self.caps)
                text_bs = 256
                with torch.no_grad():
                    for i in range(0, num_text, text_bs):
                        text = self.caps[i: min(num_text, i + text_bs)]
                        captions = []
                        for j in range(len(text)):
                            caption_tokens = clip.tokenize(text[j])
                            captions.append(caption_tokens)

                        captions = torch.cat(captions, dim=0)
                        text_embed = model.encode_text(captions.cuda())
                        self.cap_embs.append(text_embed.cpu().detach().numpy())

                    self.cap_embs = np.concatenate(self.cap_embs, axis=0)
                del text_embed
                del captions
                del model
                del self.caps
                del lines
        torch.cuda.empty_cache()

    def relabel(self):
        """Assign a label granularity to each image, class by class."""
        class_imgs = {}
        for img_path, label in self.samples:
            if label not in class_imgs.keys():
                class_imgs[label] = {'images': [], 'subord': [], 'basic': []}
                class_imgs[label]['subord'].append(trees[label][2] + 12)
                class_imgs[label]['basic'].append(trees[label][1] - 1)
            class_imgs[label]['images'].append(img_path)

        images = []
        labels = []
        fine_labels = []
        subord_labels = []
        basic_labels = []

        for key in class_imgs.keys():
            length = len(class_imgs[key]['images'])
            np.random.shuffle(class_imgs[key]['images']) # shuffle the images
            images += class_imgs[key]['images']

            sp_cnt = int(length * self.sp_proportion)
            fm_cnt = int(length * (self.fm_proportion - self.sp_proportion))
            rest = length - sp_cnt - fm_cnt

            # given labels: fine-grained / subordinate / basic
            labels += [int(key + 51)] * sp_cnt
            labels += class_imgs[key]['subord'] * fm_cnt
            labels += class_imgs[key]['basic'] * rest

            # full hierarchy, only used for evaluation
            fine_labels += [int(key)] * length
            subord_labels += class_imgs[key]['subord'] * length
            basic_labels += class_imgs[key]['basic'] * length

        return images, labels, fine_labels, subord_labels, basic_labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, given label, fine-grained, subordinate, basic) targets.
        """
        path = self.image_filenames[index]
        target = self.labels[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.texts:
            return sample, target, self.fine_labels[index], self.subord_labels[index] - 13, self.basic_labels[index], self.cap_embs[index]
        else:
            return sample, target, self.fine_labels[index], self.subord_labels[index] - 13, self.basic_labels[index]
