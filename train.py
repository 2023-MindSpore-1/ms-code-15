# -*- coding: utf-8 -*-

import mindspore as ms
from mindspore import nn, dataset, context, ops
from mindspore.ops import functional as F
import os
import argparse
from datetime import datetime
from utils.dataloader import get_loader
from utils.utils import bce_iou_loss, AvgMeter
from module.network import BCS_Net
import random
import numpy as np




class ComputeLoss(nn.Cell):
    def __init__(self, network, loss_fn1, loss_fn2):
        super(ComputeLoss, self).__init__(auto_prefix=False)
        self.network = network
        self._loss_fn = loss_fn1
        self._loss_fn2 = loss_fn2
    def construct(self, images, gts, edges):

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, g_edge = self.network(images)
        # rgb, depth, rgbd = self.network(rgb, depth)
        loss5 = self._loss_fn(lateral_map_5, gts)
        loss4 = self._loss_fn(lateral_map_4, gts)
        loss3 = self._loss_fn(lateral_map_3, gts)
        loss2 = self._loss_fn(lateral_map_2, gts)
        loss1 = self._loss_fn2(g_edge, edges)
        return loss5,loss4,loss3,loss2,loss1
        

class MultiLossTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0):
        super(MultiLossTrainOneStepCell, self).__init__(network, optimizer, sens)

    def construct(self, *inputs):
        loss5,loss4,loss3,loss2,loss1 = self.network(*inputs)
        sens1 = F.fill(loss1.dtype, loss1.shape, self.sens)
        sens2 = F.fill(loss2.dtype, loss2.shape, self.sens)
        sens3 = F.fill(loss3.dtype, loss3.shape, self.sens)
        sens4 = F.fill(loss4.dtype, loss4.shape, self.sens)
        sens5 = F.fill(loss5.dtype, loss5.shape, self.sens)
        sens = (sens1,sens2,sens3,sens4,sens5)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        
        return F.depend(loss5, self.optimizer(grads)),F.depend(loss4, self.optimizer(grads)),F.depend(loss3, self.optimizer(grads)),F.depend(loss2, self.optimizer(grads)),F.depend(loss1, self.optimizer(grads))


def train(opt):
    save_path = opt.save_path
    print('Backbone loading: Res2Net50')
    model = BCS_Net(n_class=opt.n_classes)

    BCE =  nn.BCEWithLogitsLoss()
    params = model.trainable_params()
    optimizer = nn.optim.Adam(params,
                    opt.lr,
                    eps=1e-08,
                    weight_decay=0.0)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader, iterations_epoch = get_loader(image_root, gt_root, edge_root,
                              batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=opt.num_workers)
    total_step = iterations_epoch

    # ---- start !! -----
    print("#" * 20, "start", "#" * 20)
    print('Backbone loading: Res2Net50')

    net = ComputeLoss(model, bce_iou_loss, BCE)
    train_net = MultiLossTrainOneStepCell(net, optimizer)
    model.set_train()
    for epoch in range(1, opt.epoch):
        size_rates = [0.75,1,1.25]  
        loss_record5, loss_record4, loss_record1, loss_record2, loss_record3 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_record = AvgMeter()
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                # ---- data prepare ----
                images, gts, edges = pack["image"], pack["gt"], pack["edge"]
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                images = F.squeeze(images)
                gts = F.squeeze(gts,axis=(1))
                edges = F.squeeze(edges,axis=(1))
                if rate != 1:
                    images = F.interpolate(images,sizes=(trainsize, trainsize), mode='bilinear')
                    gts = F.interpolate(gts,sizes=(trainsize, trainsize), mode='bilinear') 
                    edges = F.interpolate(edges,sizes=(trainsize, trainsize), mode='bilinear')
                
                # ---- forward ----
                loss5,loss4,loss3,loss2,loss1 = train_net(images, gts, edges)
                if rate == 1:
                    loss_record1.update(loss1.asnumpy(), opt.batchsize)
                    loss_record2.update(loss2.asnumpy(), opt.batchsize)
                    loss_record3.update(loss3.asnumpy(), opt.batchsize)
                    loss_record4.update(loss4.asnumpy(), opt.batchsize)
                    loss_record5.update(loss5.asnumpy(), opt.batchsize)
            # # ---- train logging ----
            if i % 20 == 0 or i == total_step:
                 print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-edge: {:.4f}, lateral-2: {:.4f}, '
                    'lateral-3: {:.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                     format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record1.show(),
                            loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))

        os.makedirs(save_path, exist_ok=True)
        if (epoch + 1) % 10 == 0:
            ms.save_checkpoint(model, save_path + 'bcsnet-%d' % (epoch + 1))
            print('[Saving Snapshot:]', save_path + 'bcsnet-%d.ckpt' % (epoch + 1))

def seed_mindspore(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200,
                        help='epoch number')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8,
                        help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='set the size of training sample')
    parser.add_argument('--gpu_device', type=str, default="0",
                        help='choose which GPU device you want to use')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers in dataloader. In windows, set num_workers=0')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='binary segmentation when n_classes=1')
    parser.add_argument('--train_path', type=str,
                        default='./Dataset2/TrainingSet')
    parser.add_argument('--save_path', type=str, default='./checkpoints/save_weights_multiloss/',
                        help='If you use custom save path')
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= opt.gpu_device
    ms.set_context(device_target="GPU")
    seed_mindspore()
    train(opt)

