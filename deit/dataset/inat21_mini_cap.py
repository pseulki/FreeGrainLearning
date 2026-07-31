"""iNat21-Mini under the free-grain setting (iNat21-F).

The label granularity of each training image comes from the annotation file.
The given label is encoded as
    0-272     -> basic level only
    273-1375  -> up to subordinate level
    1376+     -> fine-grained level
"""
from typing import Optional, Callable, Any, Tuple, List, Union
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import clip
from transformers import AutoTokenizer
import re


def truncate_text(hf_tokenizer, text, max_tokens=75):
    tokens = hf_tokenizer.tokenize(text)
    truncated_tokens = tokens[:max_tokens-3]
    cleaned_text = hf_tokenizer.convert_tokens_to_string(truncated_tokens)
    cleaned_text = cleaned_text.replace("</w>", " ")
    cleaned_text = re.sub(r"\s+([,.'])", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text


class iNat21MiniDataset(Dataset):
    def __init__(self,
                 root,
                 is_train: bool = True,
                 transform=None,
                 texts: str = None,
                 clip_model: str = "ViT-B/32",):
        self.transform = transform
        self.is_train = is_train
        self.texts = texts

        self.img_path = []
        self.basic_label_list = []
        self.subord_label_list = []
        self.class_label_list = []
        self.labels = []

        if is_train:
            txt = os.path.join('data/inat21-F-train.txt')
            with open(txt) as f:
                for line in f:
                    self.img_path.append(os.path.join(root, line.split()[0]))
                    self.basic_label_list.append(int(line.split()[1]))
                    self.subord_label_list.append(int(line.split()[2]))
                    self.class_label_list.append(int(line.split()[3]))
                    self.labels.append(int(line.split()[4]))
        else:
            txt = os.path.join('data/inat21-F-val.txt')
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
                hf_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

                for i in range(len(self.img_path)):
                    id = "/".join(self.img_path[i].split('/')[-3:])
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

    def __len__(self):
        return len(self.class_label_list)

    def __getitem__(self, index):
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
