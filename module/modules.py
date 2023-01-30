import mindspore
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore import ops as P
from mindspore import Parameter, Tensor
from mindspore.ops import functional as F
    
def RC(F, A):
    return F * A + F

class Upsample(nn.Cell):
    r"""
        from https://toscode.gitee.com/mindspore/mindspore/pulls/43041
    """

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        """Initialize Upsample."""
        super(Upsample, self).__init__()
        self.mode = mode
        if align_corners is None:
            align_corners = False
        if mode == "nearest":
            self.upsample = P.ResizeNearestNeighbor(size=size, align_corners=align_corners)
        elif mode == "linear":
            if align_corners:
                coordinate_transformation_mode = "align_corners"
            else:
                coordinate_transformation_mode = "asymmetric"
            self.upsample = P.image_ops.ResizeLinear1D(coordinate_transformation_mode=coordinate_transformation_mode)
            self.size = Tensor(size, dtype=mstype.int32)
        elif mode == "bilinear":
            self.upsample = P.ResizeBilinear(size=size, align_corners=align_corners)
        elif mode == "bicubic":
            self.upsample = P.image_ops.ResizeBicubic(align_corners=align_corners)
            self.size = Tensor(size, dtype=mstype.int32)
        elif mode == "trilinear":
            self.upsample = P.nn_ops.UpsampleTrilinear3D(output_size=size, align_corners=align_corners)
        else:
            raise TypeError("Only 'nearest', 'linear', 'bilinear', 'bicubic', and 'trilinear' are supported")

    def construct(self, x):
        if self.mode == "linear" or self.mode == "bicubic":
            return self.upsample(x, self.size)
        return self.upsample(x)

class BasicConv2d(nn.Cell):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, pad_mode="pad", has_bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BAUnit(nn.Cell):
    def __init__(self, num_att_maps):
        super(BAUnit, self).__init__()
        dim = 64
        self.conv_1 = nn.SequentialCell(nn.Conv2d(num_att_maps, dim, kernel_size=1, has_bias=False), nn.ReLU())
        self.conv_2 = nn.SequentialCell(nn.Conv2d(dim, dim, kernel_size=3, padding=1, pad_mode="pad", has_bias=False), nn.ReLU())
        self.conv_3 = nn.Conv2d(dim, 1, kernel_size=1, has_bias=False)
    
    def construct(self, concat_att_maps):
        fusion_att_maps = self.conv_3(self.conv_2(self.conv_1(concat_att_maps)))
        return fusion_att_maps

class SGU(nn.Cell):
    def __init__(self, channel, n_class):
        super(SGU, self).__init__()
        self.relu = nn.ReLU()
        # self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_1 = BasicConv2d(channel, channel, 3, padding=1,stride=2)
        self.conv_2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_3 = nn.SequentialCell(nn.Conv2d(channel, 16, kernel_size=1, has_bias=False), nn.ReLU())
        self.conv_4 = nn.SequentialCell(nn.Conv2d(16, 16, kernel_size=3, padding=1, pad_mode="pad", has_bias=False), nn.ReLU())
        self.conv_5 = nn.Conv2d(16, 1, kernel_size=1, has_bias=False)
    def construct(self, x1, x2, x3):
        # x = self.upsample(self.conv_2(x1))
        x = F.interpolate(self.conv_2(x1), scales = (1.,1.,2.,2.), mode='bilinear')
        x += self.conv_2(x2)
        x += self.conv_1(x3)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        return x

class GCM(nn.Cell):
    def __init__(self, in_channels, squeeze_ratio=8):
        super(GCM, self).__init__()
        inter_channels = in_channels // squeeze_ratio
        self.conv_q = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.delta = Parameter(Tensor([0],mindspore.float32))

        self.reshape = P.Reshape()
        self.tranpose = P.Transpose()        
        self.batmatmul = P.BatchMatMul()
        self.softmax = P.Softmax(axis=-1)
    def construct(self, ftr):
        B, C, H, W = ftr.shape
        P = H * W

        ftr_q = self.conv_q(ftr)
        ftr_q = self.tranpose(self.reshape(ftr_q, (B, -1, P)), (0, 2, 1)) # [B, P, C']

        ftr_k = self.conv_k(ftr)
        ftr_k = self.reshape(ftr_k, (B, -1, P)) # [B, C', P]

        gcm = self.softmax(self.batmatmul(ftr_q, ftr_k)) # column-wise softmax, [B, P, P]

        ftr_v = self.conv_v(ftr)
        ftr_v = self.reshape(ftr_v, (B, -1, P))

        G = self.reshape(self.batmatmul(ftr_v, gcm), (B, C, H, W))

        return self.delta*(G*ftr) + ftr


class SpatialAttention(nn.Cell):
    """
    from https://gitee.com/mindspore/models/blob/master/research/cv/CBAM/src/model.py
    SpatialAttention: Different from the channel attention module, the spatial attention module focuses on the
    "where" of the information part as a supplement to the channel attention module.
    """
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, pad_mode='pad', has_bias=False)
        self.concat = P.Concat(axis=1)
        self.sigmod = nn.Sigmoid()
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.max_pool = P.ReduceMax(keep_dims=True)

    def construct(self, x):
        avg_out = self.reduce_mean(x, 1)
        max_out = self.max_pool(x, 1)
        x = self.concat((avg_out, max_out))
        x = self.conv1(x)

        return self.sigmod(x)

    
class ASPP(nn.Cell):
    def __init__(self, in_channel):
        super(ASPP,self).__init__()
        depth = in_channel//2
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6,  pad_mode="pad", dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, pad_mode="pad", dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, pad_mode="pad", dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def construct(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = Upsample(size=size, mode='bilinear')(image_features)
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(P.concat((image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18), axis=1))
        return net
        
class AGGC(nn.Cell):
    def __init__(self, in_channels, use_pyramid):
        super(AGGC, self).__init__()
        self.use_pyramid = use_pyramid
        self.gcm = GCM(in_channels)
        self.sau = SpatialAttention()
        if self.use_pyramid:
            self.aspp = ASPP(in_channels)
         
    def construct(self, ftr, edge):
        att = self.sau(ftr) + edge
        ftr = RC(ftr,att)
        ftr_gcm = self.gcm(ftr)
        if self.use_pyramid:
            ftr_aspp = self.aspp(ftr_gcm)
            return ftr_aspp
        else:
            return ftr_gcm



class RFB_modified(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU()
        self.branch0 = BasicConv2d(in_channel, out_channel, 1)
        
        self.branch1 = nn.SequentialCell(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 0, 1, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 1, 0, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.SequentialCell(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 0, 2, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 2, 0, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.SequentialCell(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 0, 3, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 3, 0, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(P.concat((x0, x1, x2, x3),  axis=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x