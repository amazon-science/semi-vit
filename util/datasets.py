# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import PIL
import pandas as pd
import numpy as np

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ImageFolderWithIndex(ImageFolder):

    def __init__(self, root, indexs=None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)
        if indexs is not None:
            self.samples = [self.samples[i] for i in indexs]
            self.targets = [self.targets[i] for i in indexs]
            self.imgs = self.samples


def print_transform(transform, name):
    print("Transform ({}) = ".format(name))
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    trainindex = None
    if is_train:
        if args.trainindex is not None:
            print("load index from {}".format(args.trainindex))
            index_info = os.path.join(args.data_path, 'indexes', args.trainindex)
            index_info = pd.read_csv(index_info)
            trainindex = index_info['Index'].tolist()
        elif 0.0 < args.anno_percent < 1.0:
            print("random sampling {} percent of data".format(args.anno_percent * 100))
            base_dataset = datasets.ImageFolder(root)
            trainindex, _ = x_u_split(
                base_dataset.targets, args.anno_percent, len(base_dataset.classes))

    dataset = ImageFolderWithIndex(root, trainindex, transform=transform)
    assert len(dataset.class_to_idx) == args.nb_classes

    print(dataset)

    return dataset


def build_dataset_ssl(args):
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    # labeled and unlabeled data splitting
    if args.trainindex_x is not None and args.trainindex_u is not None:
        print("load index from {}/{}".format(args.trainindex_x, args.trainindex_u))
        index_info_x = os.path.join(args.data_path, 'indexes', args.trainindex_x)
        index_info_u = os.path.join(args.data_path, 'indexes', args.trainindex_u)
        index_info_x = pd.read_csv(index_info_x)
        trainindex_x = index_info_x['Index'].tolist()
        index_info_u = pd.read_csv(index_info_u)
        trainindex_u = index_info_u['Index'].tolist()
    else:
        print("random sampling {} percent of data".format(args.anno_percent * 100))
        base_dataset = datasets.ImageFolder(traindir)
        trainindex_x, trainindex_u = x_u_split(
            base_dataset.targets, args.anno_percent, len(base_dataset.classes))

    # data transforms
    transform_weak = build_transform_weak(args)
    transform_strong = build_transform_strong(args)
    transform_u = TwoCropsTransform(transform_weak, transform_strong)
    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)
    print_transform(transform_weak, "train weak")
    print_transform(transform_strong, "train strong")
    print_transform(transform_train, "train")
    print_transform(transform_val, "val")

    train_dataset_x = ImageFolderWithIndex(traindir, trainindex_x, transform=transform_train)
    train_dataset_u = ImageFolderWithIndex(traindir, trainindex_u, transform=transform_u)
    val_dataset = ImageFolder(valdir, transform=transform_val)

    assert len(train_dataset_x.class_to_idx) == args.nb_classes
    print("# class = %d, # labeled data = %d, # unlabeled data = %d" % (
        args.nb_classes, len(train_dataset_x.imgs), len(train_dataset_u.imgs)))

    return train_dataset_x, train_dataset_u, val_dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_transform_weak(args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # this should always dispatch to transforms_imagenet_train
    auto_augment = args.aa if args.weak_aa else None
    if not args.weak_no_aug:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=auto_augment,
            interpolation='bicubic',
            re_prob=0.,
            mean=mean,
            std=std,
        )
        return transform
    else:
        t = []
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


def build_transform_strong(args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # this should always dispatch to transforms_imagenet_train
    reprob = 0. if args.strong_no_re else args.reprob
    transform = create_transform(
        input_size=args.input_size,
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation='bicubic',
        re_prob=reprob,
        re_mode=args.remode,
        re_count=args.recount,
        mean=mean,
        std=std,
    )
    return transform


class TwoCropsTransform:
    """Take two random crops of one image."""

    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        out1 = self.transform1(x)
        out2 = self.transform2(x)
        return out1, out2

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        names = ['transform1', 'transform2']
        for idx, t in enumerate([self.transform1, self.transform2]):
            format_string += '\n'
            t_string = '{0}={1}'.format(names[idx], t)
            t_string_split = t_string.split('\n')
            t_string_split = ['    ' + tstr for tstr in t_string_split]
            t_string = '\n'.join(t_string_split)
            format_string += '{0}'.format(t_string)
        format_string += '\n)'
        return format_string


def x_u_split(labels, percent, num_classes):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        label_per_class = max(1, round(percent * len(idx)))
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    print('labeled_idx ({}): {}, ..., {}'.format(len(labeled_idx), labeled_idx[:5], labeled_idx[-5:]))
    print('unlabeled_idx ({}): {}, ..., {}'.format(len(unlabeled_idx), unlabeled_idx[:5], unlabeled_idx[-5:]))
    return labeled_idx, unlabeled_idx
