# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2021/1/16 20:35

"""
File Description

"""

from PIL import Image
import os, json
import os.path
import numpy as np
import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class INAT19(VisionDataset):
    base_folder = 'inaturalist19'
    train_list = 'train2019.json'
    test_list = 'val2019.json'

    def __init__(self, root, train=True, transform=None, target_transform=None):

        super(INAT19, self).__init__(root, transform=transform,
                                     target_transform=target_transform)

        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        file_path = os.path.join(self.root, self.base_folder, downloaded_list)
        with open(file_path, "r") as f:
            all_info = json.load(f)

        for image, label in zip(all_info['images'], all_info['annotations']):
            if image['id'] != label['id']:
                raise ValueError
            self.data.append(image['file_name'])
            self.targets.append(label['category_id'])

        self.classes = [c['name'] for c in all_info['categories']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.open(os.path.join(self.root, self.base_folder, img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)