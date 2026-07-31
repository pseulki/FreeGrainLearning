"""FGVC-Aircraft under the free-grain setting (Aircraft-F).

The label granularity of each training image is drawn according to
`sp_proportion` / `fm_proportion`, so that a sample is supervised at the
fine-grained (variant), subordinate (family) or basic (manufacturer) level.
The given label is encoded as
    0-29    -> basic level only
    30-99   -> up to subordinate level
    100-199 -> fine-grained level
"""
from typing import Optional, Callable, Any, Tuple, List, Union
from pathlib import Path
import os

import numpy as np
import torch
from torchvision.datasets import VisionDataset
import PIL.Image

import clip
from transformers import AutoTokenizer
import re

from data.air_get_tree_target import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def truncate_text(hf_tokenizer, text, max_tokens=75):
    tokens = hf_tokenizer.tokenize(text)
    truncated_tokens = tokens[:max_tokens-3]
    cleaned_text = hf_tokenizer.convert_tokens_to_string(truncated_tokens)
    cleaned_text = cleaned_text.replace("</w>", " ")
    cleaned_text = re.sub(r"\s+([,.'])", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text


class FGVCAircraft_Hier(VisionDataset):
    def __init__(self,
                 root: Union[str, Path],
                 is_train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 random_number: int = 0,
                 sp_proportion=0.1,
                 fm_proportion=0.5,
                 texts: str = None,
                 clip_model: str = "ViT-B/32",):
        super(FGVCAircraft_Hier, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform)

        np.random.seed(random_number)
        self.sp_proportion = sp_proportion
        self.fm_proportion = fm_proportion
        self.texts = texts

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
                hf_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

                for i in range(len(self.image_filenames)):
                    id = "/".join(self.image_filenames[i].split('/')[-1:])
                    self.caps.append(cap_dic[id])

                self.cap_embs = []
                num_text = len(self.caps)
                text_bs = 256
                with torch.no_grad():
                    for i in range(0, num_text, text_bs):
                        text = self.caps[i: min(num_text, i + text_bs)]
                        captions = []
                        for j in range(len(text)):
                            shorten_text = truncate_text(hf_tokenizer, text[j]) # due to clip input token limit
                            caption_tokens = clip.tokenize(shorten_text)
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
                del hf_tokenizer
        torch.cuda.empty_cache()

    def relabel(self):
        """Assign a label granularity to each image, class by class."""
        class_imgs = {}
        for img_path, label in zip(self._image_files, self._labels):
            if label not in class_imgs.keys():
                class_imgs[label] = {'images': [], 'subord': [], 'basic': []}
                class_imgs[label]['subord'].append(trees[label][1] + 29)
                class_imgs[label]['basic'].append(trees[label][2] - 1)
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
            labels += [int(key + 100)] * sp_cnt
            labels += class_imgs[key]['subord'] * fm_cnt
            labels += class_imgs[key]['basic'] * rest

            # full hierarchy, only used for evaluation
            fine_labels += [int(key)] * length
            subord_labels += class_imgs[key]['subord'] * length
            basic_labels += class_imgs[key]['basic'] * length

        return images, labels, fine_labels, subord_labels, basic_labels

    def __len__(self) -> int:
        return len(self.labels)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, given label, fine-grained, subordinate, basic) targets.
        """
        path = self.image_filenames[index]
        target = self.labels[index]

        sample = PIL.Image.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.texts:
            return sample, target, self.fine_labels[index], self.subord_labels[index] - 30, self.basic_labels[index], self.cap_embs[index]
        else:
            return sample, target, self.fine_labels[index], self.subord_labels[index] - 30, self.basic_labels[index]
