# -*- coding: utf-8 -*-

import mindspore.nn as nn
from module.backbone.Res2Net import res2net50
from module.modules import *
from mindspore import ops
from mindspore.ops import functional as F
import mindspore as ms
import numpy as np

class BCS_Net(nn.Cell):
    def __init__(self, n_class=1):
        super(BCS_Net, self).__init__()
        ch = [512, 1024, 2048]

        self.AGGC1 = AGGC(ch[0],True)
        self.AGGC2 = AGGC(ch[1],False)
        self.AGGC3 = AGGC(ch[2],False)
        self.resnet = res2net50(pretrained=True)

        self.cat4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.cat4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.cat4_conv3 = BasicConv2d(256, 64, kernel_size=1)
        self.cat4_conv4 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.cat4_conv5 = BasicConv2d(64, n_class, kernel_size=1)

        self.cat3_conv1 = BasicConv2d(1088, 256, kernel_size=1)
        self.cat3_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.cat3_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.cat3_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.cat3_conv5 = BasicConv2d(256, n_class, kernel_size=1)

        self.cat2_conv1 = BasicConv2d(768, 256, kernel_size=1)
        self.cat2_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.cat2_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.cat2_conv4 = BasicConv2d(256, n_class, kernel_size=1)


        self.cat2_conv1 = BasicConv2d(512, 256, kernel_size=1)
        self.cat2_conv3 = BasicConv2d(256, 64, kernel_size=1)
        self.cat2_conv4 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.cat2_conv4 = BasicConv2d(64, n_class, kernel_size=1)

        # ---- edge branch ----
        self.bau = BAUnit(256)

        self.rfb2 = RFB_modified(512, 32)
        self.rfb3 = RFB_modified(1024, 32)
        self.rfb4 = RFB_modified(2048, 32)
        self.sgu = SGU(32, n_class)

        self.sigmoid =  nn.Sigmoid()
        self.relu = nn.ReLU()

    def construct(self, x):
        f = self.resnet.conv1_0(x)
        f = self.resnet.bn1_0(f)
        f = self.resnet.relu(f)
        f = self.resnet.conv1_1(f)
        f = self.resnet.bn1_1(f)
        f = self.resnet.relu(f)
        f = self.resnet.conv1_2(f)

        f = self.resnet.bn1(f)
        f = self.resnet.relu(f)
        if self.resnet.res_base:
            f = self.resnet.pad(f)
        f = self.resnet.maxpool(f)

        f1 = self.resnet.layer1(f)
        f2 = self.resnet.layer2(f1)
        f3 = self.resnet.layer3(f2)
        f4 = self.resnet.layer4(f3)

        # ---- Boundary Attention ----
        boundary_featrues = self.bau(f1)
        S_b = self.sigmoid(boundary_featrues)

        boundary_map = F.interpolate(boundary_featrues,
                               scales = (1.,1.,4.,4.),
                               mode='bilinear')

        rfb_x2 = self.rfb2(f2)
        rfb_x3 = self.rfb3(f3)
        rfb_x4 = self.rfb4(f4)

        sgu_feat = self.sgu(rfb_x4, rfb_x3, rfb_x2)
        S_s = self.sigmoid(sgu_feat)
        s_global = F.interpolate(sgu_feat,
                                      scales = (1.,1.,16.,16.),
                                      mode='bilinear')

        # ---- AGGC0 ----
        S_b = F.interpolate(S_b, scales = (1.,1.,0.5,0.5), mode='bilinear')
        x_g_2 = self.AGGC1(f2,S_b)

        # ---- AGGC2 ----
        S_b = F.interpolate(S_b, scales = (1.,1.,0.5,0.5), mode='bilinear')
        x_g_3 = self.AGGC2(f3,S_b)

        # ---- AGGC2 ----
        S_b = F.interpolate(S_b, scales = (1.,1.,0.5,0.5), mode='bilinear')
        x_g_4 = self.AGGC3(f4,S_b)
 
        lateral_global4 = F.interpolate(S_s, scales = (1.,1.,0.5,0.5), mode='bilinear')

        x_g_4_m = x_g_4 * lateral_global4
        x_g_4 = x_g_4 + x_g_4_m
        x_g_4 = self.relu(self.cat4_conv1(x_g_4))
        x_g_4 = self.relu(self.cat4_conv2(x_g_4))
        x_g_4 = self.relu(self.cat4_conv3(x_g_4))
        x_g_4 = self.relu(self.cat4_conv4(x_g_4))
        cat4_feat = self.cat4_conv5(x_g_4)

        lateral_map_4 = F.interpolate(cat4_feat,
                                      scales = (1.,1.,32.,32.),
                                      mode='bilinear')

        x_g_3_cat_4 = ops.concat((x_g_3, F.interpolate(x_g_4, scales = (1.,1.,2.,2.), mode='bilinear')), axis=1)
  
        lateral_global3 = F.interpolate(S_s, scales = (1.,1.,1.,1.), mode='bilinear')

        x_g_3_cat_4_m = x_g_3_cat_4 * lateral_global3
        x_g_3_cat_4 = x_g_3_cat_4 + x_g_3_cat_4_m
        x_g_3_cat_4 = self.relu(self.cat3_conv1(x_g_3_cat_4))
        x_g_3_cat_4 = self.relu(self.cat3_conv2(x_g_3_cat_4))
        x_g_3_cat_4 = self.relu(self.cat3_conv3(x_g_3_cat_4))
        x_g_3_cat_4 = self.relu(self.cat3_conv4(x_g_3_cat_4))
        cat3_feat = self.cat3_conv5(x_g_3_cat_4)
        lateral_map_3 = F.interpolate(cat3_feat,
                                      scales = (1.,1.,16.,16.),
                                      mode='bilinear')

        x_g_2_cat_3 = ops.concat((x_g_2, F.interpolate(x_g_3_cat_4, scales = (1.,1.,2.,2.), mode='bilinear')), axis=1)
        lateral_global2 = F.interpolate(S_s, scales = (1.,1.,2.,2.), mode='bilinear')
       
        x_g_2_cat_3_m = x_g_2_cat_3 * lateral_global2
        x_g_2_cat_3 = x_g_2_cat_3 + x_g_2_cat_3_m
        x_g_2_cat_3 = self.relu(self.cat2_conv1(x_g_2_cat_3))
        x_g_2_cat_3 = self.relu(self.cat2_conv2(x_g_2_cat_3))
        x_g_2_cat_3 = self.relu(self.cat2_conv3(x_g_2_cat_3))
        cat2_feat = self.cat2_conv4(x_g_2_cat_3)
        lateral_map_2 = F.interpolate(cat2_feat,
                                      scales = (1.,1.,8.,8.),
                                      mode='bilinear')
        return s_global,lateral_map_4,lateral_map_3, lateral_map_2, boundary_map

if __name__ == '__main__':
    ms.set_context(device_target="GPU")
    x = ms.Tensor(np.ones([1,3,224, 224]).astype(np.float32))
    print("x====================",x.shape)   
    model = BCS_Net()
    s_global,lateral_map_4,lateral_map_3, lateral_map_2, boundary_map = model(x)
    print(s_global.shape)
