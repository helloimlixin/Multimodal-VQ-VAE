#  Copyright (c) 2023. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import os
import torch

from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class FLIRDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.rgb_imgs_train = os.listdir(os.path.join(self.root_dir, 'images_rgb_train'))
        self.thermal_imgs_train = os.listdir(os.path.join(self.root_dir, 'images_thermal_train'))
        self.rgb_imgs_train.sort()
        self.thermal_imgs_train.sort()

        self.rgb_imgs_val = os.listdir(os.path.join(self.root_dir, 'images_rgb_val'))
        self.thermal_imgs_val = os.listdir(os.path.join(self.root_dir, 'images_thermal_val'))
        self.rgb_imgs_val.sort()
        self.thermal_imgs_val.sort()

    def __len__(self):
        if self.train:
            return len(self.rgb_imgs_train)
        else:
            return len(self.rgb_imgs_val)

    def __getitem__(self, idx):
        rgb_img_path, thermal_img_path = None, None
        if self.train:
            rgb_img_path = os.path.join(self.root_dir, 'images_rgb_train', self.rgb_imgs_train[idx])
            thermal_img_path = os.path.join(self.root_dir, 'images_thermal_train', self.thermal_imgs_train[idx])
        else:
            rgb_img_path = os.path.join(self.root_dir, self.rgb_imgs_val[idx])
            thermal_img_path = os.path.join(self.root_dir, self.thermal_imgs_val[idx])

        rgb_img = read_image(rgb_img_path)
        thermal_img = read_image(thermal_img_path)
        rgb_img = F.convert_image_dtype(rgb_img, dtype=torch.float)
        thermal_img = F.convert_image_dtype(thermal_img, dtype=torch.float)

        if self.transform:
            rgb_img = self.transform(rgb_img)
            thermal_img = self.transform(thermal_img)

        return {'rgb_image': rgb_img, 'thermal_image': thermal_img}

