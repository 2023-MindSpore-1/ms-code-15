# -*- coding: utf-8 -*-

import os
import random
import numpy as np
from PIL import Image
import mindspore
from mindspore import dataset
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms import Compose

class COVIDDatasetGenerator():
    def __init__(self, image_root, gt_root, edge_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        if len(edge_root) != 0:
            self.edge_flage = True
            self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.png')]
            self.edges = sorted(self.edges)
        else:
            self.edge_flage = False

        self.filter_files()
        self.size = len(self.images)

        self.img_transform = Compose([
            vision.Resize((self.trainsize, self.trainsize)),
            vision.ToTensor(),
            vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], is_hwc=False)])
        self.gt_transform = Compose([
            vision.Resize((self.trainsize, self.trainsize)),
            vision.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        if self.edge_flage:
            edge = self.binary_loader(self.edges[index])
            edge = self.gt_transform(edge)
            return image, gt, edge
        else:
            return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):
    dataset_generator = COVIDDatasetGenerator(image_root, gt_root, edge_root, trainsize)
    data_loader = dataset.GeneratorDataset(
                    dataset_generator, ["image", "gt", "edge"], shuffle=True, num_parallel_workers=4)        
    data_loader = data_loader.batch(batchsize)
    iterations_epoch = data_loader.get_dataset_size()
    train_iterator = data_loader.create_dict_iterator()
    return train_iterator, iterations_epoch

class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.transform = Compose([
            vision.Resize((self.testsize, self.testsize)),
            vision.ToTensor(),
            vision.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225],is_hwc=False)])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1

        return image, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
