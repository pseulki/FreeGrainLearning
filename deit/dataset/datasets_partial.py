# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import imagenet_f
import imagenet_f_seeds
import imagenet_f_seeds_cap

# import birds_partial 
# import birds_partial_seeds

# import birds_real
# import birds_real_seeds

# import aircraft_partial
# import aircraft_partial_seeds

# import inat21_mini_seeds
# import inat21_mini_seeds_cap
# import inat21_mini_cap



def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'IMNET-F':
        nb_classes=[505, 127, 20]
        dataset = imagenet_f.ImageNetHier(
                 args.data_path, 
                 is_train,
                 transform=transform,
                 texts=args.texts,
        )
    elif args.data_set == 'IMNET-F-SUPERPIXEL':
        nb_classes=[505, 127, 20]
        dataset = imagenet_f_seeds.ImageNetHier( #_cap
                 args.data_path, 
                 is_train,
                 transform=transform,
                 mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
                n_segments=args.num_superpixels,
                compactness=10.0,
                blur_ops=None,
                scale_factor=1.0,
        )

    elif args.data_set == 'IMNET-F-SUPERPIXEL-CAP':
        nb_classes=[505, 127, 20]
        dataset = imagenet_f_seeds_cap.ImageNetHier( #_cap
                 args.data_path, 
                 is_train,
                 transform=transform,
                 mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
                n_segments=args.num_superpixels,
                compactness=10.0,
                blur_ops=None,
                scale_factor=1.0,
                texts=args.texts,
                path_yn=args.path_yn
        )

    
    elif args.data_set == 'BIRD-SYN':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = birds_partial.ImageFolder(
            root,
            transform=transform,
            is_train=is_train,
            random_number=args.random_seed,
            sp_proportion=args.sp_proportion,
            fm_proportion=args.fm_proportion,
            texts=args.texts,
        )
        nb_classes = [200, 38, 13]
    elif args.data_set == 'BIRD-SYN-SUPERPIXEL':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = birds_partial_seeds.ImageFolder(
            root,
            transform=transform,
            is_train=is_train,
            random_number=args.seed,
            sp_proportion=args.sp_proportion,
            fm_proportion=args.fm_proportion,
            texts=args.texts,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = [200, 38, 13]

    elif args.data_set == 'AIR-SYN':
        dataset = aircraft_partial.FGVCAircraft_Hier(
            args.data_path,
            transform=transform,
            is_train=is_train,
            random_number=args.random_seed,
            sp_proportion=args.sp_proportion,
            fm_proportion=args.fm_proportion,
            texts=args.texts,
        )

        nb_classes = [100, 70, 30]

    elif args.data_set == 'AIR-SYN-SUPERPIXEL':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = aircraft_partial_seeds.FGVCAircraft_Hier(
            args.data_path,
            transform=transform,
            is_train=is_train,
            random_number=args.seed,
            sp_proportion=args.sp_proportion,
            fm_proportion=args.fm_proportion,
            texts=args.texts,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes =[100, 70, 30]

        
    elif args.data_set == 'BIRD-F':
        dataset = birds_real.BirdRealDataset(
            args.data_path,
            transform=transform,
            is_train=is_train,
            texts=args.texts,
        )
        nb_classes = [200, 38, 13]
    
    elif args.data_set == 'BIRD-F-SUPERPIXEL':
        dataset = birds_real_seeds.BirdRealDataset(
            args.data_path,
            transform=transform,
            is_train=is_train,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
            texts=args.texts,
        )
        nb_classes = [200, 38, 13]
    
    elif args.data_set == 'INAT21-MINI-HIER-SUPERPIXEL':
        dataset = inat21_mini_seeds.iNat21MiniDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            mean=[0.466, 0.471, 0.380],
            std=[0.195, 0.194, 0.192],
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = [10000, 1103, 273]

    elif args.data_set == 'INAT21-MINI-HIER-SUPERPIXEL-CAP':
        dataset = inat21_mini_seeds_cap.iNat21MiniDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            mean=[0.466, 0.471, 0.380],
            std=[0.195, 0.194, 0.192],
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
            texts=args.texts,
        )
        nb_classes = [10000, 1103, 273]

    elif args.data_set == 'INAT21-MINI-HIER-CAP':
        dataset = inat21_mini_cap.iNat21MiniDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            texts=args.texts,
        )
        nb_classes = [10000, 1103, 273]


    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if 'INAT' in args.data_set:
        t.append(transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
