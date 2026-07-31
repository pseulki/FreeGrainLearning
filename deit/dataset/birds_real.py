"""CUB-200-2011 under the real free-grain setting (CUB-Real).

Unlike `birds_partial.py`, the label granularity is not synthesized: it comes from
the annotation file, which records the level each image was actually labeled at.
The given label is encoded as
    0-12    -> basic level only
    13-50   -> up to subordinate level
    51-250  -> fine-grained level
"""
from typing import Optional, Callable, Any, Tuple, List, Union
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import clip


class BirdRealDataset(Dataset):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 is_train: bool = True,
                 texts: str = None,
                 clip_model: str = "ViT-B/32",):

        self.root = root
        self.transform = transform
        self.is_train = is_train
        self.texts = texts

        self.img_path = []
        self.basic_label_list = []
        self.subord_label_list = []
        self.class_label_list = []
        self.labels = []

        if is_train:
            txt = os.path.join('data/cub-Real-train.txt')
            with open(txt) as f:
                for line in f:
                    self.img_path.append(os.path.join(root, line.split()[0]))
                    self.basic_label_list.append(int(line.split()[1]))
                    self.subord_label_list.append(int(line.split()[2]))
                    self.class_label_list.append(int(line.split()[3]))
                    self.labels.append(int(line.split()[4]))
        else:
            txt = os.path.join('data/cub-Real-val.txt')
            with open(txt) as f:
                for line in f:
                    self.img_path.append(os.path.join(root, line.split()[0]))
                    self.basic_label_list.append(int(line.split()[1]))
                    self.subord_label_list.append(int(line.split()[2]))
                    self.class_label_list.append(int(line.split()[3]))

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
                for i in range(len(self.img_path)):
                    id = "/".join(self.img_path[i].split('/')[-2:])
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

    def __len__(self):
        return len(self.class_label_list)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, given label, fine-grained, subordinate, basic) targets.
        """
        path = self.img_path[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        if self.is_train:
            if self.texts:
                return sample, self.labels[index], self.class_label_list[index], self.subord_label_list[index], self.basic_label_list[index], self.cap_embs[index]
            else:
                return sample, self.labels[index], self.class_label_list[index], self.subord_label_list[index], self.basic_label_list[index]
        else:
            return sample, self.class_label_list[index], self.subord_label_list[index], self.basic_label_list[index]
