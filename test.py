# -*- coding: utf-8 -*-
import os
import argparse
from scipy import misc
import time
import random
import numpy as np
from module.network import BCS_Net as Network
from utils.dataloader import test_dataset
import cv2
import mindspore
from mindspore import nn, dataset, context

def seed_mindspore(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--data_path', type=str, default='./Dataset2/TestingSet/',
                        help='Path to test data')
    parser.add_argument('--pth_path', type=str, default='./checkpoints/save_weights/model_BCS.ckpt',
                        help='Path to weights file. ')
    parser.add_argument('--save_path', type=str, default='./Results/BCSNET/',
                        help='Path to save the predictions. ')
    parser.add_argument('--gpu_device', type=str, default="0",
                        help='choose which GPU device you want to use')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']= opt.gpu_device
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    seed_mindspore()

    net = Network()
    param_dict = mindspore.load_checkpoint(opt.pth_path)
    mindspore.load_param_into_net(net, param_dict)
    model = mindspore.Model(net)
    image_root = '{}/Imgs/'.format(opt.data_path)
    test_loader = test_dataset(image_root, opt.testsize)
    os.makedirs(opt.save_path, exist_ok=True)

    for i in range(test_loader.size):
        image, name = test_loader.load_data()
        lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, g_edge = model.predict(mindspore.Tensor(image))
        res = lateral_map_1     
        # res = F.upsample(res, size=(ori_size[1],ori_size[0]), mode='bilinear', align_corners=False)
        res = res.sigmoid().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = res.asnumpy()
        cv2.imwrite(opt.save_path + name, res*255)
        # misc.imsave(opt.save_path + name, res)
    print('Test Done!')

if __name__ == "__main__":
    inference()
