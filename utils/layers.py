import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
https://github.com/pytorch/pytorch/issues/12207
'''
class UpsampleDeterministic(nn.Module):
    def __init__(self, upscale=2):
        super(UpsampleDeterministic, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        '''
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,self.upscale*h,self.upscale*w)
        '''
        return x[:, :, :, None, :, None].expand(-1, -1, -1, self.upscale, -1,
                                                self.upscale).reshape(
                                                    x.size(0), x.size(1),
                                                    x.size(2) * self.upscale,
                                                    x.size(3) * self.upscale)

def GroupNrom(x, gamma, beta, G=16):
    # [N, C, H, W]
    results = 0
    eps = 1e-5
    x = np.reshape(x,(x.shape[0],G, x.shape[1]/G, x.shape[2], x.shape[3]))

    x_mean = np.mean(x, axis=(2,3,4),keepdims=True)
    x_var = np.var(x, axis=(2,3,4),keepdims=True)
    x_normalized = (x-x_mean)/np.sqrt(x_var+eps)
    results = gamma * x_normalized + beta
    return results

'''
rfb
https://github.com/ruinmessi/RFBNet/blob/master/models/RFB_Net_vgg.py
'''
class BasicConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(
            out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):
    '''
    [rfb]
    filters = 128
    stride = 1 or 2
    scale = 1.0
    '''
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes,
                      2 * inter_planes,
                      kernel_size=1,
                      stride=stride),
            BasicConv(2 * inter_planes,
                      2 * inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=visual,
                      dilation=visual,
                      relu=False))
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes,
                      2 * inter_planes,
                      kernel_size=(3, 3),
                      stride=stride,
                      padding=(1, 1)),
            BasicConv(2 * inter_planes,
                      2 * inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=visual + 1,
                      dilation=visual + 1,
                      relu=False))
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            BasicConv((inter_planes // 2) * 3,
                      2 * inter_planes,
                      kernel_size=3,
                      stride=stride,
                      padding=1),
            BasicConv(2 * inter_planes,
                      2 * inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=2 * visual + 1,
                      dilation=2 * visual + 1,
                      relu=False))

        self.ConvLinear = BasicConv(6 * inter_planes,
                                    out_planes,
                                    kernel_size=1,
                                    stride=1,
                                    relu=False)
        self.shortcut = BasicConv(in_planes,
                                  out_planes,
                                  kernel_size=1,
                                  stride=stride,
                                  relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class BasicRFB_small(nn.Module):
    '''
    [rfbs]
    filters = 128
    stride=1 or 2
    scale = 1.0
    '''
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_small, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes,
                      inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      relu=False))
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes,
                      inter_planes,
                      kernel_size=(3, 1),
                      stride=1,
                      padding=(1, 0)),
            BasicConv(inter_planes,
                      inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=3,
                      dilation=3,
                      relu=False))
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes,
                      inter_planes,
                      kernel_size=(1, 3),
                      stride=stride,
                      padding=(0, 1)),
            BasicConv(inter_planes,
                      inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=3,
                      dilation=3,
                      relu=False))
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3,
                      kernel_size=(1, 3),
                      stride=1,
                      padding=(0, 1)),
            BasicConv((inter_planes // 4) * 3,
                      inter_planes,
                      kernel_size=(3, 1),
                      stride=stride,
                      padding=(1, 0)),
            BasicConv(inter_planes,
                      inter_planes,
                      kernel_size=3,
                      stride=1,
                      padding=5,
                      dilation=5,
                      relu=False))

        self.ConvLinear = BasicConv(4 * inter_planes,
                                    out_planes,
                                    kernel_size=1,
                                    stride=1,
                                    relu=False)
        self.shortcut = BasicConv(in_planes,
                                  out_planes,
                                  kernel_size=1,
                                  stride=stride,
                                  relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


'''
GCNet -ContextBlock
'''


class ContextBlock(nn.Module):
    '''
    cfg [gcblock]
    ratio=16

    '''
    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


'''
sknet
'''


class SKConv(nn.Module):
    def __init__(self, features, M, G=32, r=16, stride=1, L=32):
        """ 
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        Cfg:
            branch=num
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        # print("D:", d)
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            # print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            # print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


'''
https://blog.csdn.net/qq_33270279/article/details/103034316
'''


#新增Res2net模块y
class Bottle2neck(nn.Module):
    expansion = 2  #通道压缩比，降低模型的计算量。

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 baseWidth=26,
                 scale=4,
                 stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality #输入的特征数
            planes: output channel dimensionality #1*1卷积通道压缩后的通道数，该模块最终的输出：planes*expansion
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3 #单一尺度上的3*3卷积的通道数量，控住不同尺度间，不同特征的重复利用的程度。
            scale: number of scale. #分为几个尺度
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes,
                               width * scale,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width,
                          width,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.leaky = nn.LeakyReLU(0.1, inplace=True)
        self.linear = nn.LeakyReLU(1.0, inplace=True)  #纯线性激活函数，后期会做解释。
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.leaky(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.leaky(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.linear(out)

        return out


#新增Triangle模块
class Triangle(nn.Module):
    expansion = 2  #通道压缩比，降低模型的计算量。

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 baseWidth=16,
                 scale=4,
                 stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality #输入的特征数
            planes: output channel dimensionality #1*1卷积通道压缩后的通道数，该模块最终的输出：planes*expansion
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3 #单一尺度上的3*3卷积的通道数量，控住不同尺度间，不同特征的重复利用的程度。
            scale: number of scale. #分为几个尺度
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Triangle, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes,
                               width * scale,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width,
                          width,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.leaky = nn.LeakyReLU(0.1, inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.leaky(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        return out


class DWConv(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_plane,
                                    out_channels=in_plane,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_plane)
        self.point_conv = nn.Conv2d(in_channels=in_plane,
                                    out_channels=out_plane,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


'''
https://zhuanlan.zhihu.com/p/100064668
'''

# class RFBModule(nn.Module):
#     def __init__(self, channels, reduction=2, leakyReLU=False):
#         super(RFBModule, self).__init__()
#         self.module_list = nn.ModuleList()
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1x1 = nn.Sequential(nn.Conv2d(channels, channels, 1),
#                                      nn.BatchNorm2d(channels))
#         self.branch_1 = nn.Conv2d(channels,
#                                   channels // reduction,
#                                   3,
#                                   1,
#                                   padding=1)

#         self.leakyReLU1 = nn.LeakyReLU(inplace=True)

#         self.branch_2 = nn.Conv2d(channels,
#                                   channels // reduction,
#                                   3,
#                                   padding=2,
#                                   dilation=2)
#         self.leakyReLU2 = nn.LeakyReLU(inplace=True)
#         self.branch_3 = nn.Conv2d(channels,
#                                   channels // reduction,
#                                   3,
#                                   padding=3,
#                                   dilation=3)
#         self.leakyReLU3 = nn.LeakyReLU(inplace=True)

#         self.fusion = nn.Sequential(
#             nn.Conv2d(channels // reduction * 3, channels, 1),
#             nn.BatchNorm2d(channels))

#     def forward(self, x):
#         x = self.conv1x1(x)
#         x_1 = self.leakyReLU1(self.branch_1(x))
#         x_2 = self.leakyReLU2(self.branch_2(x))
#         x_3 = self.leakyReLU3(self.branch_3(x))

#         # print(x_1.shape, x_2.shape, x_3.shape)
#         x_f = torch.cat([x_1, x_2, x_3], 1)
#         return self.relu(self.fusion(x_f))
'''
https://github.com/DingXiaoH/ACNet
'''

class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]

class ACBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size),
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=True,
                                        padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=(kernel_size,
                                                      kernel_size),
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1,
                               center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border,
                               center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            # print(square_outputs.size())
            # return square_outputs
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.size())
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            # print(horizontal_outputs.size())
            return square_outputs + vertical_outputs + horizontal_outputs


def Conv2dBNReLU(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 use_original_conv=False):
    # if use_original_conv or kernel_size == 1 or kernel_size == (1, 1):
    #     return super(ACNetBuilder,
    #                  self).Conv2dBNReLU(in_channels=in_channels,
    #                                     out_channels=out_channels,
    #                                     kernel_size=kernel_size,
    #                                     stride=stride,
    #                                     padding=padding,
    #                                     dilation=dilation,
    #                                     groups=groups,
    #                                     padding_mode=padding_mode,
    #                                     use_original_conv=True)
    # else:
    se = nn.Sequential()
    deploy = False
    se.add_module(
        'acb',
        ACBlock(in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                padding_mode=padding_mode,
                deploy=deploy))
    se.add_module('relu', nn.ReLU())
    return se


'''
https://github.com/tahaemara/LiteSeg
'''


class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   1,
                                   1,
                                   0,
                                   1,
                                   1,
                                   bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ASPP(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP, self).__init__()
        self.rate = rate
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
            #self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, bias=False,padding=1)
            self.conv1 = SeparableConv2d(planes, planes, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.ReLU()

            #self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
            #                         stride=1, padding=padding, dilation=rate, bias=False)
        self.atrous_convolution = SeparableConv2d(inplanes, planes,
                                                  kernel_size, 1, padding,
                                                  rate)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        #x = self.relu(x)
        if self.rate != 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


'''
https://github.com/fregu856/deeplabv3
'''

# class ASPP(nn.Module):
#     def __init__(self, num_classes):
#         super(ASPP, self).__init__()

#         self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)

#         self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

#         self.conv_3x3_1 = nn.Conv2d(512,
#                                     256,
#                                     kernel_size=3,
#                                     stride=1,
#                                     padding=6,
#                                     dilation=6)
#         self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

#         self.conv_3x3_2 = nn.Conv2d(512,
#                                     256,
#                                     kernel_size=3,
#                                     stride=1,
#                                     padding=12,
#                                     dilation=12)
#         self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

#         self.conv_3x3_3 = nn.Conv2d(512,
#                                     256,
#                                     kernel_size=3,
#                                     stride=1,
#                                     padding=18,
#                                     dilation=18)
#         self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
#         self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

#         self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)  # (1280 = 5*256)
#         self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

#         self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)

#     def forward(self, feature_map):
#         # (feature_map has shape (batch_size, 512, h/16, w/16))
#         # (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16.
#         # If self.resnet instead is ResNet18_OS8 or ResNet34_OS8,
#         # it will be (batch_size, 512, h/8, w/8))

#         feature_map_h = feature_map.size()[2]  # (== h/16)
#         feature_map_w = feature_map.size()[3]  # (== w/16)

#         # print("=====", feature_map.shape)
#         out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
#         # (shape: (batch_size, 256, h/16, w/16))
#         out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
#         # (shape: (batch_size, 256, h/16, w/16))
#         out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
#         # (shape: (batch_size, 256, h/16, w/16))
#         out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))
#         # (shape: (batch_size, 256, h/16, w/16))
#         # print('-------')

#         out_img = self.avg_pool(feature_map)
#         # (shape: (batch_size, 512, 1, 1))
#         out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
#         # (shape: (batch_size, 256, 1, 1))
#         out_img = F.upsample(out_img,
#                              size=(feature_map_h, feature_map_w),
#                              mode="bilinear")
#         # (shape: (batch_size, 256, h/16, w/16))
#         # print('?'*5)
#         out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
#         # (shape: (batch_size, 1280, h/16, w/16))
#         out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
#         # (shape: (batch_size, 256, h/16, w/16))
#         out = self.conv_1x1_4(out)
#         # (shape: (batch_size, num_classes, h/16, w/16))

#         return out
'''
https://github.com/Lextal/pspnet-pytorch
'''


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1),
                                    out_features,
                                    kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.upsample(input=stage(feats), size=(h, w), mode='bilinear')
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)



class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] *
                                           (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x.mul_(torch.sigmoid(x))


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x.mul_(F.softplus(x).tanh())
