from typing import Dict, Any

import torchvision.transforms.v2 as tfms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from torchvision.transforms.v2 import functional as F, Transform


def get_resize_transform(h, w):
    transform = tfms.Compose([
        tfms.ConvertImageDtype(dtype=torch.float32),
        tfms.Resize((h, w), antialias=True)
    ])
    return transform


def get_train_transform_for_cls_agnostic(args):
    transform = tfms.Compose([
        tfms.ToImageTensor(),
        tfms.ConvertImageDtype(dtype=torch.float32),
        tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        tfms.RandomHorizontalFlip(),
        tfms.RandomResizedCrop(size=(args.input_size, args.input_size), scale=(0.7, 1.), ratio=(0.9, 1.1), antialias=True)
    ])
    return transform


def get_test_transform_for_cls_agnostic(args=None):
    transform = tfms.Compose([
        tfms.ToImageTensor(),
        tfms.ConvertImageDtype(dtype=torch.float32),
        tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        PadToDevisible16(fill=0)
    ])
    return transform


def get_train_transform_for_pseudo(args):
    transform = tfms.Compose([
        tfms.ToImageTensor(),
        tfms.ConvertImageDtype(dtype=torch.float32),
        tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        tfms.RandomHorizontalFlip(),
        tfms.RandomResizedCrop(size=(args.input_size, args.input_size), scale=(0.7, 1.), ratio=(0.9, 1.1), antialias=True)
    ])
    return transform


def get_test_transform_for_pseudo(args=None):
    transform = tfms.Compose([
        tfms.ToImageTensor(),
        tfms.ConvertImageDtype(dtype=torch.float32),
        tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        PadToDevisible16(fill=0)
    ])
    return transform


def get_image_transform_for_classifier():
    transform = tfms.Compose([
        tfms.ToImageTensor(),
        tfms.ConvertImageDtype(dtype=torch.float32),
        tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return transform


def get_train_transform(args):
    transform = tfms.Compose([
        tfms.ToImageTensor(),
        tfms.ConvertImageDtype(dtype=torch.float32),
        tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        tfms.RandomHorizontalFlip(),
        # do not crop or distort too much
        tfms.RandomResizedCrop(size=(args.input_size, args.input_size), scale=(0.7, 1.), ratio=(0.9, 1.1), antialias=True)
    ])
    return transform


def get_train_focused_transform(args):
    transform = tfms.Compose([
        tfms.ToImageTensor(),
        tfms.ConvertImageDtype(dtype=torch.float32),
        tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        tfms.RandomHorizontalFlip(),
        PadToSquare(fill=0),
        tfms.RandomResizedCrop(size=(args.input_size, args.input_size), scale=(0.7, 1.), ratio=(0.9, 1.1),
                               antialias=True)
    ])
    return transform


def get_test_focused_transform(args):
    transform = tfms.Compose([
        tfms.ToImageTensor(),
        tfms.ConvertImageDtype(dtype=torch.float32),
        tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        PadToSquare(fill=0),
        tfms.Resize(size=(args.input_size, args.input_size), antialias=True)
    ])
    return transform


class PadToSquare(Transform):
    """ pad the input to have height and width equal """
    def __init__(self, fill=0, padding_mode='constant'):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        h, w = inpt.shape[-2:]
        if h == w:
            return inpt
        dim_diff = abs(h - w)
        pad_size = dim_diff // 4
        pad = (pad_size, 0, pad_size, 0) if h >= w else (0, pad_size, 0, pad_size)
        return F.pad(inpt, padding=pad, fill=self.fill, padding_mode=self.padding_mode)


class PadToDevisible16(Transform):
    """ pad the input to have height and width devisible by 16 """
    def __init__(self, fill=0, padding_mode='constant'):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        h, w = inpt.shape[-2:]
        if h % 16 == 0 and w % 16 == 0:
            return inpt
        pad = (0, 0, 16 - (w % 16), 16 - (h % 16))
        return F.pad(inpt, padding=pad, fill=self.fill, padding_mode=self.padding_mode)
