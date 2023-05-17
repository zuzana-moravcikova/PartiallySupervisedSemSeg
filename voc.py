from typing import Any, Callable, Dict, Optional, Tuple, List
import warnings
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.voc import verify_str_arg, download_and_extract_archive, DATASET_YEAR_DICT
from torchvision import datapoints as dp
from torchvision import disable_beta_transforms_warning
import torchvision.transforms as tfms
from transforms import get_image_transform_for_classifier
disable_beta_transforms_warning()

import torch

import os

# ROOT = "/datagrid/TextSpotter/moravzu5/datasets/VOC"
ROOT = "C:/cvut/datasets/VOC2"
CATS = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
         'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
         'train', 'tvmonitor', 'ignore']

CATS_DICT = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5:'bottle',
                     6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
                     13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep',
                     18: 'sofa', 19: 'train', 20: 'tvmonitor'}

CLASSES = [x for x in range(len(CATS) - 1)] + [255]
COLORMAP = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
            (224, 224, 192)]
CAT2IDX = {cat: idx for idx, cat in enumerate(CATS)}
IDX2CAT = {idx: cat for idx, cat in enumerate(CATS)}
CLS2IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX2CLS = {idx: cls for idx, cls in enumerate(CLASSES)}
CAT2CLS = {cat: CLASSES[idx] for idx, cat in enumerate(CATS)}
CLS2CAT = {cls: cat for cat, cls in CAT2CLS.items()}
CLS2COLOR = {cls: COLORMAP[idx] for idx, cls in enumerate(CLASSES)}
CAT2COLOR = {cat: COLORMAP[idx] for idx, cat in enumerate(CATS)}


# categories split into two sets, the 1st, 3nd, 5th, ... most common classes are in A, 2nd, 4th, ... in B
CATS_A = ['person', 'car', 'cat', 'bottle', 'tvmonitor', 'train', 'pottedplant', 'boat', 'horse', 'sheep']
CATS_B = ['dog', 'chair', 'bird', 'aeroplane', 'bicycle', 'diningtable', 'motorbike', 'sofa', 'bus', 'cow']

class _VOCBase(VisionDataset):
    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str

    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.year = verify_str_arg(year, "year", valid_values=[str(yr) for yr in range(2007, 2013)])

        valid_image_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_image_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_image_sets)

        key = "2007-test" if year == "2007" and image_set == "test" else year
        dataset_year_dict = DATASET_YEAR_DICT[key]

        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]

        base_dir = dataset_year_dict["base_dir"]
        voc_root = os.path.join(self.root, base_dir)

        if download:
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, self._TARGET_DIR)
        self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]

        assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)


class VOCClassification(_VOCBase):
    """
    Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Classification Dataset.

        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
            image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
                ``year=="2007"``, can also be ``"test"``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            transforms (callable, optional): A function/transform that takes input sample and its target as entry
                and returns a transformed version.
        """

    _SPLITS_DIR = "Main"
    _TARGET_FILE_EXT = ".txt"

    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "train",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        VisionDataset.__init__(self, root, transforms, transform, target_transform)
        if year == "2007-test":
            if image_set == "test":
                warnings.warn(
                    "Acessing the test image set of the year 2007 with year='2007-test' is deprecated. "
                    "Please use the combination year='2007' and image_set='test' instead."
                )
                year = "2007"
            else:
                raise ValueError(
                    "In the test image set of the year 2007 only image_set='test' is allowed. "
                    "For all other image sets use year='2007' instead."
                )
        self.year = year

        valid_image_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_image_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_image_sets)

        key = "2007-test" if year == "2007" and image_set == "test" else year
        dataset_year_dict = DATASET_YEAR_DICT[key]

        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]

        if download:
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)

        base_dir = dataset_year_dict["base_dir"]
        self.voc_root = os.path.join(self.root, base_dir)
        if not os.path.isdir(self.voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        self.image_dir = os.path.join(self.voc_root, "JPEGImages")

        # global index is the official one, local is for this dataset implementation
        self.loc2globcls = {idx: cls for idx, cls in enumerate(CLASSES[1:-1])}
        self.glob2loccls = {cls: idx for idx, cls in enumerate(CLASSES[1:-1])}

        if not os.path.exists(self.cls_file):
            splits_dir = os.path.join(self.voc_root, "ImageSets", self._SPLITS_DIR)
            print(f'Classification GT {self.cls_file} does not exist, it will be created now.')
            self.generate_cls_file(splits_dir, self.image_set, self.glob2loccls, self.cls_file)
            print('Classification GT has been created successfully!')
        self.images, self.targets = self.load_dataset()

        assert len(self.images) == len(self.targets)

    @property
    def cls_file(self):
        return os.path.join(self.voc_root, "ImageSets", self._SPLITS_DIR, f'cls_{self.image_set}.npy')


    def generate_cls_file(self, splits_dir, image_set, glob2loc, cls_file_path):
        imnames = []
        target_dict = {}

        for cat in CATS[1:-1]:
            loc_cls = glob2loc[CAT2CLS[cat]]
            txt_path = os.path.join(splits_dir, f"{cat}_{image_set}.txt")

            with open(txt_path) as f:
                split_lines = [line.split() for line in f.readlines()]
                cat_imnames = [imname for imname, cls in split_lines if int(cls) > 0]
                for im in cat_imnames:
                    im_label = target_dict[im] if im in target_dict.keys() else np.zeros(len(CATS) - 2)
                    im_label[loc_cls] = 1
                    target_dict[im] = im_label
                imnames += cat_imnames

        im_set = sorted(set(imnames))
        targets = np.array([target_dict[im] for im in im_set])
        np.save(cls_file_path, {'array': targets, 'names': np.array(im_set), 'dict': target_dict})

    def get_cls_distribution(self):
        cls_array = np.load(self.cls_file, allow_pickle=True).item()['array']
        cls_im_counts = cls_array.sum(axis=0)
        sorted_idxs = sorted([x for x in range(len(cls_im_counts))], key=lambda x: cls_im_counts[x], reverse=True)
        for idx in sorted_idxs:
            print(f'{CATS[idx + 1]}: {cls_im_counts[idx]}')


        print([CATS[idx + 1] for idx in sorted_idxs[::2]])
        print([CATS[idx + 1] for idx in sorted_idxs[1::2]])
        # for cls, count in zip(CATS[1:-1], cls_im_counts):
        #     print(f'{cls}: {count}')

    def load_dataset(self):
        data = np.load(self.cls_file, allow_pickle=True).item()
        return data['names'], data['array']

    def __getitem__(self, item):
        im_path = os.path.join(self.image_dir, self.images[item] + ".jpg")
        im = Image.open(im_path)
        target = self.targets[item]
        if self.transform:
            im = self.transform(im)
        if self.target_transform:
            target = self.target_transform(target)
        meta = {'image_dir': self.image_dir, 'name': self.images[item], 'clsfile': self.cls_file}
        return im, target, meta

    def __len__(self):
        return len(self.images)


class VOCClassificationSub(VOCClassification):
    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "trainval",
            sub: str = "dog&cat",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super(VOCClassificationSub, self).__init__(root, year, image_set, download, transform, target_transform, transforms)
        self.sub, self.cats = sort_and_verify_sub(sub)
        # used ie. for indexing of the general class file
        self.rel_loc_idxs = [self.glob2loccls[CAT2CLS[cat]] for cat in self.cats]
        self.sub2loccls = {idx: loc_idx for idx, loc_idx in enumerate(self.rel_loc_idxs)}
        self.loc2subcls = {loc: sub for sub, loc in self.sub2loccls.items()}
        self.sub2globcls = {sub: self.loc2globcls[loc] for sub, loc in enumerate(self.rel_loc_idxs)}
        self.glob2subcls = {glob: sub for sub, glob in self.sub2globcls.items()}
        self.images, self.targets = self.generate_subdataset()

    def generate_subdataset(self):
        # load the annotation for the whole dataset
        data = np.load(self.cls_file, allow_pickle=True).item()
        names_cls, array_cls = data['names'], data['array']

        # take just the relevant subset
        loc_clses = [self.glob2loccls[CAT2CLS[cat]] for cat in self.cats]
        sub_idxs = np.unique(np.where(array_cls[:, loc_clses])[0])
        im_names = names_cls[sub_idxs]
        cls_targets = array_cls[sub_idxs][:, loc_clses]
        return im_names, cls_targets


class VOCSegmentationSub(_VOCBase):
    """
    A pytorch implementation of the pascal semantic segmentation sub-dataset.
     Use when you only need a subset of the classes.
    """

    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "trainval",
            sub: str = "dog&cat",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            joint_transforms: Optional[Callable] = None,
    ):

        self.common_init(root, year, image_set, sub, download, transform, target_transform, joint_transforms)

        if not os.path.exists(self.cls_file):
            print(f'Classification GT {self.cls_file} does not exist, it will be created now.')
            self.generate_cls_file()
            print('Classification GT has been created successfully!')
        self.images, self.cls_targets = self.generate_subdataset()
        self.name2idx = {name: idx for idx, name in enumerate(self.images)}

    def common_init(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "trainval",
            sub: str = "dog&cat",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            joint_transforms: Optional[Callable] = None,):


        VisionDataset.__init__(self, root, transform=transform, target_transform=target_transform)
        self.joint_transforms = joint_transforms
        if year == "2007-test":
            if image_set == "test":
                warnings.warn(
                    "Acessing the test image set of the year 2007 with year='2007-test' is deprecated. "
                    "Please use the combination year='2007' and image_set='test' instead."
                )
                year = "2007"
            else:
                raise ValueError(
                    "In the test image set of the year 2007 only image_set='test' is allowed. "
                    "For all other image sets use year='2007' instead."
                )
        self.year = year

        valid_image_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_image_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_image_sets)

        key = "2007-test" if year == "2007" and image_set == "test" else year
        dataset_year_dict = DATASET_YEAR_DICT[key]

        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]

        base_dir = dataset_year_dict["base_dir"]
        self.voc_root = os.path.join(self.root, base_dir)

        if download:
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)

        if not os.path.isdir(self.voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.image_dir = os.path.join(self.voc_root, "JPEGImages")
        self.mask_dir = os.path.join(self.voc_root, "SegmentationClass")

        self.sub, self.cats = sort_and_verify_sub(sub)

        # global index is the official one, local is for this dataset implementation
        # (excludes bg and ignore from indexing)
        self.loc2globcls = {idx: cls for idx, cls in enumerate(CLASSES[1:-1])}
        self.glob2loccls = {cls: idx for idx, cls in enumerate(CLASSES[1:-1])}
        # used ie. for indexing of the general class file
        self.rel_loc_idxs = [self.glob2loccls[CAT2CLS[cat]] for cat in self.cats]
        self.sub2loccls = {idx: loc_idx for idx, loc_idx in enumerate(self.rel_loc_idxs)}
        self.loc2subcls = {loc: sub for sub, loc in self.sub2loccls.items()}
        self.sub2globcls = {sub: self.loc2globcls[loc] for sub, loc in enumerate(self.rel_loc_idxs)}
        self.glob2subcls = {glob: sub for sub, glob in self.sub2globcls.items()}

    @property
    def cls_file(self):
        return os.path.join(self.voc_root, "ImageSets", "Segmentation", f'cls_{self.image_set}.npy')

    def cls_file_all(self):
        return os.path.join(self.voc_root, "ImageSets", "Segmentation", f'cls_trainval.npy')

    def __getitem__(self, item):
        name = self.images[item]
        im, full_mask = self.load_im_and_mask(name)
        sub_mask = self.create_submask(full_mask, self.cls_targets[item])
        if self.transform:
            im = self.transform(im)
        if self.target_transform:
            sub_mask = self.target_transform(sub_mask)
        if self.joint_transforms:
            im, sub_mask = self.joint_transforms(im, sub_mask)
        return im, sub_mask

    def create_submask(self, full_mask, cls_target):
        #  create a new mask with only the sub-dataset classes and the sub-dataset indices
        in_sub_clses = np.where(cls_target)[0]
        sub_mask = np.zeros_like(full_mask)

        for sub_cls in in_sub_clses:
            glob_cls = self.sub2globcls[sub_cls]
            cls_mask = get_cls_mask(full_mask, glob_cls, glob_cls)
            sub_mask += cls_mask
        return sub_mask

    def generate_subdataset(self):
        # load the annotation for the whole dataset
        data = np.load(self.cls_file, allow_pickle=True).item()
        names_cls, dict_cls, array_cls = data['names'], data['dict'], data['array']

        # take just the relevant subset
        loc_clses = [self.glob2loccls[CAT2CLS[cat]] for cat in self.cats]
        sub_idxs = np.unique(np.where(array_cls[:, loc_clses])[0])
        im_names = names_cls[sub_idxs]
        # take only the im name without path and ext. to look up in the target dict
        cls_targets = array_cls[sub_idxs][:, loc_clses]
        return im_names, cls_targets

    def generate_whole_dataset(self, cls_file_name):
        # load the annotation for the whole dataset
        data = np.load(cls_file_name, allow_pickle=True).item()
        names_cls, dict_cls, array_cls = data['names'], data['dict'], data['array']
        return names_cls, array_cls

    def generate_cls_file(self):
        target_dict = {}

        splits_dir = os.path.join(self.voc_root, "ImageSets", "Segmentation")
        split_f = os.path.join(splits_dir, self.image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f), "r") as f:
            names = [x.strip() for x in f.readlines()]

        target_array = np.zeros((len(names), len(CATS) - 2))
        for idx, name in enumerate(names):
            mask_path = os.path.join(self.mask_dir, name + ".png")
            mask = Image.open(mask_path)
            clses = mask2cls(mask)
            loc_clses = [self.glob2loccls[cls] for cls in clses if cls not in [0, 255]]
            target_array[idx][loc_clses] = 1
            target_dict[name] = target_array[idx]

        np.save(self.cls_file, {'array': target_array, 'names': np.array(names), 'dict': target_dict})

    def load_im(self, name):
        im_path = os.path.join(self.image_dir, name + ".jpg")
        im = np.array(Image.open(im_path))
        return im

    def load_im_and_mask(self, name):
        im_path = os.path.join(self.image_dir, name + ".jpg")
        mask_path = os.path.join(self.mask_dir, name + ".png")
        im = np.array(Image.open(im_path))
        full_mask = np.array(Image.open(mask_path))
        return im, full_mask

    def prepare_img_for_cam_generation(self, name):
        img = self.load_im(name)
        img = torch.unsqueeze(torch.tensor(img).permute(2, 0, 1), 0)
        transform = get_image_transform_for_classifier()
        img = transform(img)
        return img

    def load_im_cam_mask(self, name, cls_targets, cam_generator):
        im, full_mask = self.load_im_and_mask(name)

        img_for_cam = self.prepare_img_for_cam_generation(name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        orig_cam = cam_generator.generate_cams(img_for_cam, cls_targets, device=device)

        h, w, _ = im.shape

        to_pil = tfms.ToPILImage()
        cam = to_pil(orig_cam)
        cam = cam.resize((w, h))
        cam = np.array(cam)

        cam = np.expand_dims(cam, axis=-1)

        return im, full_mask, cam

    def resize_cam(self, cam, height, width):
        to_pil = tfms.ToPILImage()
        cam = to_pil(cam)
        cam = cam.resize((width, height))
        cam = np.array(cam)
        cam = np.expand_dims(cam, axis=-1)
        return cam


class VOCSeg2ClsSub(VOCSegmentationSub):
    """
    Like the classification dataset, but on the segmentation image subset
    """
    def __getitem__(self, item):
        name = self.images[item]
        cls_target = self.cls_targets[item]
        im = self.load_im(name)
        meta = {'image_dir': self.image_dir, 'name': self.images[item], 'clsfile': self.cls_file}
        return im, cls_target, meta


    def load_im(self, name):
        im_path = os.path.join(self.image_dir, name + ".jpg")
        im = np.array(Image.open(im_path))
        return im


class VOCSegmentationSubBi(VOCSegmentationSub):
    """
    Segmentation dataset which always return an image, a single binary class and the class idx for the class.
    """

    def __getitem__(self, item):
        name, cls_targets = self.images[item], self.cls_targets[item]
        cls_targets = subset_clses_to_loc(cls_targets, self.glob2loccls)
        im, full_mask = self.load_im_and_mask(name)
        sub_mask = self.create_submask(full_mask, cls_targets)
        if self.transform:
            im = self.transform(im)
        if self.target_transform:
            sub_mask = self.target_transform(sub_mask)
        if self.joint_transforms:
            im, sub_mask = self.joint_transforms(im, sub_mask)
        return im, sub_mask, cls_targets

    def generate_subdataset(self):
        # Unroll so thatEach image with N classes gets converted to N images with 1 class
        # load the annotation for the whole dataset
        data = np.load(self.cls_file, allow_pickle=True).item()
        names_cls, dict_cls, array_cls = data['names'], data['dict'], data['array']

        # take just the relevant subset
        loc_clses = [self.glob2loccls[CAT2CLS[cat]] for cat in self.cats]
        sub_idxs = np.unique(np.where(array_cls[:, loc_clses])[0])

        im_names_collapsed = names_cls[sub_idxs]
        cls_targets_collapsed = array_cls[sub_idxs][:, loc_clses]

        im_names, cls_targets = [], []
        for im_name, target in zip(im_names_collapsed, cls_targets_collapsed):
            pos_idxs = np.argwhere(target > 0).flatten()
            cls_count = len(pos_idxs)
            unrolled_targets = np.zeros((cls_count, len(target)))
            unrolled_targets[np.arange(cls_count), pos_idxs.flatten()] = 1
            im_names.extend(cls_count * [im_name])
            cls_targets.append(unrolled_targets)
        cls_targets = np.vstack(cls_targets)
        return im_names, cls_targets


class VOCSegmentationSubBiCam(VOCSegmentationSubBi):
    """
    Segmentation dataset which always return an input, CAM, CAM for other classes,
     a single binary class and the class idx for the class.
    """

    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "trainval",
            sub: str = "dog&cat",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            joint_transforms: Optional[Callable] = None,
            cam_generator: Optional[Callable] = None,
            mode: str = "both",
            device: str = "cuda",
            other_cams: bool = True,
            other_cams_max=True,
            other_cams_present=True
    ):
        super().__init__(root, year, image_set, sub, download, transform, target_transform, joint_transforms)
        print("=> VOCSegmentationSubBiCam creating")

        self.cam_generator = cam_generator
        self.mode = mode
        self.other_cams = other_cams
        self.other_cams_max = other_cams_max
        self.negative_cams_present = other_cams_present

        self.cams = []
        self.other_cams = []

        for i, name in enumerate(self.images):
            img = self.prepare_img_for_cam_generation(name)
            img = img.to(device)
            targets = subset_clses_to_loc(self.cls_targets[i], self.sub2loccls)
            cls = np.argmax(targets)
            cam, other_cam = self.cam_generator.get_cams(img, cls, device=device,
                                                         pred_cams=other_cams_present,
                                                         take_max=other_cams_max)
            self.cams.append(cam)
            self.other_cams.append(other_cam)

    def __getitem__(self, item):
        name, cls_targets, cam, other_cam = self.images[item], self.cls_targets[item], self.cams[item], \
        self.other_cams[item]
        im, full_mask = self.load_im_and_mask(name)

        h, w, _ = im.shape

        cam = self.resize_cam(cam, h, w)
        other_cam = self.resize_cam(other_cam, h, w)

        sub_mask = self.create_submask(full_mask, cls_targets)
        sub_mask = sub_mask / sub_mask.max()

        cls_targets = subset_clses_to_loc(cls_targets, self.sub2loccls)

        im = dp.Image(im.transpose(2, 0, 1))
        sub_mask = dp.Mask(sub_mask[None], dtype=float)
        cam = dp.Mask(cam.transpose(2, 0, 1))
        other_cam = dp.Mask(other_cam.transpose(2, 0, 1))

        if self.joint_transforms:
            im, sub_mask, cam, other_cam = self.joint_transforms(im, sub_mask, cam, other_cam)

        cam = dp.Image(cam).float()
        other_cam = dp.Image(other_cam).float()

        if self.mode == "both":
            im = np.concatenate((im, cam), axis=0)
            if self.other_cams:
                im = np.concatenate((im, other_cam), axis=0)

        return im, sub_mask, cam, other_cam, cls_targets, name


class SegmentationPseudo(VOCSegmentationSub):
    """
    Segmentation  dataset which always returns an image, pseudo mask, targets and image name.
    The validation set also returns the GT mask.
    The pseudo mask is generated by the class_agnostic_model.
    """

    def __init__(self,
            root: str,
            year: str = "2012",
            image_set: str = "trainval",
            pseudo_mask_transform: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            cam_generator: Optional[Callable] = None,
            device: str = "cuda",
            negative_cams: bool = False,
            class_agnostic_model=None,
            threshold: float = 0.5,
            class_agnostic_model_name=None):

        super().__init__(root=root, year=year, image_set=image_set, sub="all")

        self.cam_generator = cam_generator
        self.negative_cams = negative_cams
        self.model = class_agnostic_model
        self.model.eval()
        self.psuedo_mask_transform = pseudo_mask_transform
        self.transform = transform
        self.device = device
        self.threshold = threshold
        self.class_agnostic_model_name = class_agnostic_model_name
        self.cur_pseudo_root = os.path.join(self.voc_root, "pseudo", self.class_agnostic_model_name, image_set)

        class_ds = VOCClassificationSub(root=root, year=year, image_set="trainval", sub="all", download=False)
        val_images = VOCSegmentationSub(root=root, year=year, image_set="val", sub="all", download=False).images

        if image_set == "train":
            images = []
            cls_targets = []
            for i, name in enumerate(class_ds.images):
                if name not in val_images:
                    images.append(name)
                    cls_targets.append(class_ds.targets[i])

            self.images = images
            self.cls_targets = cls_targets

        if not os.path.exists(self.cur_pseudo_root):
            self.generate_pseudo_masks()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        name, cls_targets = self.images[item], self.cls_targets[item]
        if self.image_set == "train":
            im = self.load_im(name)
        else:
            im, gt_mask = self.load_im_and_mask(name)
            gt_mask = self.create_submask(gt_mask, cls_targets)

        pseudo_mask = np.load(os.path.join(self.cur_pseudo_root, "pseudo_preds", name + ".npy"))
        pseudo_mask = pseudo_mask.squeeze(0)
        if self.image_set == "val":
            gt_mask = dp.Mask(gt_mask[None], dtype=float)

        im = dp.Image(im.transpose(2, 0, 1))
        pseudo_mask = dp.Mask(pseudo_mask[None], dtype=float)

        if self.transform:
            if self.image_set == "val":
                im, pseudo_mask, gt_mask = self.transform(im, pseudo_mask, gt_mask)
                return im, pseudo_mask, gt_mask, cls_targets, name
            else:
                im, pseudo_mask = self.transform(im, pseudo_mask)
                return im, pseudo_mask, cls_targets, name

    def generate_pseudo_masks(self):
        print("=> Generating pseudo masks. Can take a while.")
        for i, name in enumerate(self.images):
            cls_targets = self.cls_targets[i]
            im = self.load_im(name)
            h, w, _ = im.shape
            im_for_cam_gen = self.prepare_img_for_cam_generation(name)
            pseudo_preds = np.zeros((len(cls_targets) + 1, h, w))

            for i, cls in enumerate(cls_targets):
                if cls == 1:
                    cur_target = np.zeros(len(cls_targets))
                    cur_target[i] = 1
                    pred = self.pseudo_mask_for_one_cls(cur_target, im, im_for_cam_gen, h, w)
                    pseudo_preds[i + 1] = pred

            pseudo_preds[0] = 1 - np.sum(pseudo_preds, axis=0)
            pseudo_preds = pseudo_preds.argmax(axis=0)
            mask = torch.unsqueeze(torch.tensor(pseudo_preds), dim=0).int()
            mask_root = os.path.join(self.cur_pseudo_root, "pseudo_preds")

            if not os.path.exists(mask_root):
                os.makedirs(mask_root)
            np.save(os.path.join(mask_root, name + ".npy"), mask)

    def pseudo_mask_for_one_cls(self, cur_target, im, im_for_cam_gen, h, w):
        cls = cur_target.argmax()
        cam, other_cams = self.cam_generator.get_cams(im_for_cam_gen, cls, device=self.device)
        cam = self.resize_cam(cam, h, w)
        other_cams = self.resize_cam(other_cams, h, w)
        im = dp.Image(im.transpose(2, 0, 1))
        cam = dp.Mask(cam.transpose(2, 0, 1))
        other_cams = dp.Mask(other_cams.transpose(2, 0, 1))

        img, cam, other_cams = self.psuedo_mask_transform(im, cam, other_cams)
        cam = dp.Image(cam).float()
        other_cams = dp.Image(other_cams).float()
        inp = np.concatenate((img, cam, other_cams), axis=0)
        inp = torch.unsqueeze(torch.tensor(inp), dim=0).float()

        pred = self.model(inp.to(self.device))
        pred = pred[:, :, :h, :w]
        pred = torch.sigmoid(pred)
        pred = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
        return pred


class VOCSegmentationSubFgBg(VOCSegmentationSub):
    """
    Segmentation foreground/background dataset which always returns an image, a single binary class map
    and the class idx for the class.
    The image is a crop around an object.
    """

    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "trainval",
            sub: str = "dog&cat",
            download: bool = False,
            transform: Optional[Callable] = None,
            bbox_transform: Optional[Callable] = None,
    ):

        self.common_init(root, year, image_set, sub, download, transform)


        if not os.path.exists(self.cls_file):
            print(f'Classification GT {self.cls_file} does not exist, it will be created now.')
            self.generate_cls_file()
            print('Classification GT has been created successfully!')

        if not os.path.exists(self.bbox_file):
            print(f'BBOX GT {self.bbox_file} does not exist, it will be created now.')
            self.generate_bbox_file()
            print('BBOX GT has been created successfully!')
        self.images, self.bboxes, self.cls_targets = self.generate_subdataset()
        self.name2idx = {name: idx for idx, name in enumerate(self.images)}

        self.bbox_transform = bbox_transform

    @property
    def bbox_file(self):
        return os.path.join(self.voc_root, "ImageSets", "Segmentation", f"bboxes_{self.sub}_{self.image_set}.npy")

    def __getitem__(self, item):
        name, cls_targets, bbox = self.images[item], self.cls_targets[item], self.bboxes[item]
        # cls_targets = subset_clses_to_loc(cls_targets, self.sub2loccls)
        im, full_mask = self.load_im_and_mask(name)
        sub_mask = self.create_submask(full_mask, cls_targets)
        cls_targets = subset_clses_to_loc(cls_targets, self.sub2loccls)
        transformed_bbox = self.bbox_transform(bbox, im.shape)
        cropped_im, cropped_mask = crop_bbox(im, *transformed_bbox), crop_bbox(sub_mask, *transformed_bbox)
        cropped_mask = cropped_mask > 0
        # transpose because it will be converted to torch tensor, transforms assume chw
        cropped_im = dp.Image(cropped_im.transpose(2, 0, 1))
        cropped_mask = dp.Mask(cropped_mask[None], dtype=float)
        if self.transform:
            cropped_im, cropped_mask = self.transform(cropped_im, cropped_mask)
        return cropped_im, cropped_mask, cls_targets, name

    def generate_subdataset(self):
        # Unroll so that each image with N connected components gets converted to N images with 1 object
        data = np.load(self.cls_file, allow_pickle=True).item()
        bbox_dict = np.load(self.bbox_file, allow_pickle=True).item()

        names_cls, dict_cls, array_cls = data['names'], data['dict'], data['array']

        sub_idxs = np.unique(np.where(array_cls[:, self.rel_loc_idxs])[0])

        im_names, cls_targets, bboxes = [], [], []
        for im_name in names_cls[sub_idxs]:
            im_bboxes, im_bbox_clses = [], []
            if not im_name in bbox_dict:
                print('Something wrong in bbox creation')
                continue
            loc_clses_in = bbox_dict[im_name].keys() & self.rel_loc_idxs
            for cls in loc_clses_in:
                cls_bboxes = bbox_dict[im_name][cls]
                im_bboxes.extend(cls_bboxes)
                im_bbox_clses.extend([self.loc2subcls[cls]] * len(cls_bboxes))
            bboxes.extend(im_bboxes)
            unrolled_targets = np.zeros((len(im_bboxes), len(self.rel_loc_idxs)))
            unrolled_targets[np.arange(len(im_bboxes)), im_bbox_clses] = 1
            im_names.extend(len(im_bboxes) * [im_name])
            cls_targets.append(unrolled_targets)
        cls_targets = np.vstack(cls_targets)
        bboxes = np.vstack(bboxes)
        return im_names, bboxes, cls_targets

    def generate_bbox_file(self):
        # since the bbox extraction in expensive, only do this once and save it to a file
        data = np.load(self.cls_file, allow_pickle=True).item()
        names_cls, dict_cls, array_cls = data['names'], data['dict'], data['array']

        # take just the relevant subse
        # take the im idxs where one of the classes is, remove repetitions
        sub_idxs = np.unique(np.where(array_cls[:, self.rel_loc_idxs])[0])

        bbox_dict = {}
        for im_name, target in zip(names_cls[sub_idxs], array_cls[sub_idxs]):
            mask = self.load_mask(im_name)
            cls_pos_idxs = np.argwhere(target > 0).flatten()
            im_bboxes = {}
            for loc_cls in cls_pos_idxs:
                cls_mask = get_cls_mask(mask, self.loc2globcls[loc_cls], 1)
                im_bboxes[loc_cls] = bboxes_from_binary_mask(cls_mask)
                if len(im_bboxes[loc_cls]) == 0:
                    print()
            bbox_dict[im_name] = im_bboxes
        np.save(self.bbox_file, bbox_dict, allow_pickle=True)

    def load_mask(self, name):
        mask_path = os.path.join(self.mask_dir, name + ".png")
        full_mask = np.array(Image.open(mask_path))
        return full_mask



class VOCSegmentationSubFgBgCam(VOCSegmentationSubFgBg):
    """
    Segmentation foreground/background dataset which always returns an input
    (CAM if mode == 'cam' or image + CAM if mode == 'both'), a single binary class map,
    CAM of target category and the class idx for the class.
    The image is a crop around an object.
    """

    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "trainval",
            sub: str = "dog&cat",
            cam_generator=None,
            mode: str = "both",
            download: bool = False,
            transform: Optional[Callable] = None,
            bbox_transform: Optional[Callable] = None,
            device: str = "cuda",
    ):
        super(VOCSegmentationSubFgBgCam, self).__init__(root=root,
                                                         year=year,
                                                         image_set=image_set,
                                                         sub=sub,
                                                         download=download,
                                                         transform=transform,
                                                         bbox_transform=bbox_transform)

        print("=> initializing dataset")
        self.mode = mode
        self.cams = []

        for i, name in enumerate(self.images):
            img = self.load_im(name)
            img = torch.unsqueeze(torch.tensor(img).permute(2, 0, 1), 0).float() / 255
            transform = tfms.Compose([tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            transform(img)

            targets = self.cls_targets[i]
            targets = subset_clses_to_loc(targets, self.sub2loccls)
            cls = 0
            for j, target in enumerate(targets):
                if target == 1:
                    cls = j
                    break

            cam, other_cam = cam_generator.get_cams_for_all(img, cls, device=device)
            self.cams.append(cam)

    def __getitem__(self, item):
        name, cls_targets, bbox, cam = self.images[item], self.cls_targets[item], self.bboxes[item], self.cams[item]
        im, full_mask = self.load_im_and_mask(name)
        h, w, _ = im.shape

        cam = self.resize_cam(cam, h, w)

        sub_mask = self.create_submask(full_mask, cls_targets)
        cls_targets = subset_clses_to_loc(cls_targets, self.sub2loccls)
        transformed_bbox = self.bbox_transform(bbox, im.shape)

        cropped_im = crop_bbox(im, *transformed_bbox)
        cropped_mask = crop_bbox(sub_mask, *transformed_bbox)
        cropped_cam = crop_bbox(cam, *transformed_bbox)

        cropped_mask = cropped_mask > 0

        cropped_im = dp.Image(cropped_im.transpose(2, 0, 1))
        cropped_mask = dp.Mask(cropped_mask[None], dtype=float)
        cropped_cam = dp.Mask(cropped_cam.transpose(2, 0, 1))

        if self.transform:
            cropped_im, cropped_cam, cropped_mask = self.transform(cropped_im, cropped_cam, cropped_mask)

        cropped_cam = dp.Image(cropped_cam).float()

        if self.mode == "both":
            cropped_im = np.concatenate((cropped_im, cropped_cam), axis=0)

        return cropped_im, cropped_mask, cropped_cam, cls_targets, name


class VOCSegmentationSubFgBgStrict(VOCSegmentationSubFgBg):
    """
    Segmentation foreground/background dataset which always returns an image, a single binary class map
    and the class idx for the class. Strict means we make sure there is no other (annotated) object within the crop.
    The image is a crop around an object.
    """

    @property
    def bbox_file(self):
        return os.path.join(self.voc_root, "ImageSets", "Segmentation", f"bboxes_strict_{self.sub}_{self.image_set}.npy")

    def generate_bbox_file(self):
        # since the bbox extraction in expensive, only do this once and save it to a file
        data = np.load(self.cls_file, allow_pickle=True).item()
        names_cls, dict_cls, array_cls = data['names'], data['dict'], data['array']

        # take just the relevant subse
        # take the im idxs where one of the classes is, remove repetitions
        sub_idxs = np.unique(np.where(array_cls[:, self.rel_loc_idxs])[0])

        bbox_dict = {}
        for im_name, target in zip(names_cls[sub_idxs], array_cls[sub_idxs]):
            mask = self.load_mask(im_name)
            cls_pos_idxs = np.argwhere(target > 0).flatten()
            im_bboxes = {}
            for loc_cls in cls_pos_idxs:
                cls_mask = get_cls_mask(mask, self.loc2globcls[loc_cls], 1)
                other_cls_mask = cls_mask.copy()
                other_cls_mask[cls_mask] = 0
                im_bboxes[loc_cls] = bboxes_from_binary_mask(cls_mask, other_cls_mask)
                if len(im_bboxes) == 0:
                    print()
            bbox_dict[im_name] = im_bboxes
        np.save(self.bbox_file, bbox_dict, allow_pickle=True)


def bboxes_from_binary_mask(mask, exclude_mask=None):
    def bbox_from_region(region):
        Y_MIN, X_MIN, Y_MAX, X_MAX = 0, 1, 2, 3
    #     returns skimage bbox in cv2 format: [minx, miny, width, height]
        return [region.bbox[X_MIN], region.bbox[Y_MIN], region.bbox[X_MAX] - region.bbox[X_MIN], region.bbox[Y_MAX] - region.bbox[Y_MIN]]

    # find all connected components
    label_im = label(mask)
    regions = regionprops(label_im)
    bboxes = [bbox_from_region(region) for region in regions]

    # filter out by area
    bboxes = np.array([bbox for bbox in bboxes if (bbox[2] * bbox[3] > 25 and bbox[2] > 5 and bbox[3] > 5)])
    if exclude_mask:
        #     filter out bboxes that contain some excluded area
        should_exclude = []
        for bbox in bboxes:
            bbox_mask = crop_bbox(mask, *bbox)
            should_exclude += [sum(bbox_mask) > 0]
        bboxes = bboxes[should_exclude]
    return bboxes


def get_cls_mask(mask, cls, new_cls=1):
    new_mask = np.zeros_like(mask)
    idxs = np.where(mask == cls)
    if len(idxs) > 0:
        new_mask[idxs] = new_cls
    return new_mask


def sort_and_verify_sub(sub):
    if sub == 'all':
        sub = '&'.join(CATS[1:-1])
    if sub == 'A':
        sub = '&'.join(CATS_A)
    if sub == 'B':
        sub = '&'.join(CATS_B)
    verified_clses = [verify_str_arg(cls, "class in sub", CATS[1:-1]) for cls in sub.split("&")]
    sorted_cats = sorted(verified_clses)
    assert len(sorted_cats) > 1, "Only datasets with more than 1 class are supported at the moment"
    return '&'.join(sorted_cats), sorted_cats


def mask2cls(mask):
    return np.unique(mask)


def mask2color(mask):
    colour_mask = np.zeros((*mask.shape, 3))
    clses = np.unique(mask)
    for cls in clses:
        color = CLS2COLOR[cls]
        idxs = np.where(mask == cls)
        if len(idxs):
            colour_mask[idxs] = color
    return colour_mask


def crop_bbox(im, topx, topy, w, h):
    return im[topy:topy + h, topx:topx + w]

def subset_clses_to_loc(cls_targets, sub2loccls):
    out = [0 for _ in range(20)]
    for i, val in enumerate(cls_targets):
        if val == 1:
            out[sub2loccls[i]] = 1

    return torch.FloatTensor(out)


class HBBoxTransform(object):
    """
    Transform horizontal bounding box by extending/shrinking it
    """
    def __init__(self, range=(-0.1, 1.2)):
        self.range = range

    def __call__(self, bbox, im_shape):
        topx, topy, w, h = bbox
        cx, cy = np.round(topx + w / 2).astype(int), np.round(topy + h / 2).astype(int)
        im_h, im_w, _ = im_shape
        w_scale = 1 + np.random.uniform(*self.range)
        h_scale = 1 + np.random.uniform(*self.range)
        # make sure it fits into the image on all sides
        new_w_half = np.clip(0.5 * w * w_scale, 0, min(im_w - cx, cx)).astype(int)
        new_h_half = np.clip(0.5 * h * h_scale, 0, min(im_h - cy, cy)).astype(int)
        # create the new bbox
        new_topx, new_topy = cx - new_w_half, cy - new_h_half
        new_bbox = np.array([new_topx, new_topy, 2 * new_w_half, 2 * new_h_half])
        return new_bbox




