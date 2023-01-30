import numpy as np
import mindspore as ms
from mindspore import nn, dataset, context, ops
from mindspore.ops import functional as F


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return np.mean(np.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))
        #return mindspore.ops.mean(mindspore.ops.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))

def bce_iou_loss(pred, mask):
    # weit = 1 + 5 * F.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weit = 1 + 5 * F.abs(F.avg_pool2d(mask, kernel_size=31, strides=1, pad_mode='same') - mask)
    # weights = ops.OnesLike(pred)
    # pos_weight = weights
    # wbce = F.binary_cross_entropy_with_logits(pred, mask, weights,reduction='none')
    nnwbce = nn.BCEWithLogitsLoss(reduction='none')
    wbce = nnwbce(pred, mask) 
    wbce = (weit * wbce).sum(axis =(2, 3)) / weit.sum(axis=(2, 3))
    sigmoid = ops.Sigmoid()
    pred = sigmoid(pred)
    inter = ((pred * mask) * weit).sum(axis=(2, 3))
    union = ((pred + mask) * weit).sum(axis=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()