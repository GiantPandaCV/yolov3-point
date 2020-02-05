import torch.nn.functional as F

from utils.google_utils import *
from utils.parse_config import *
from utils.utils import *

ONNX_EXPORT = False
'''
https://github.com/fregu856/deeplabv3
'''


class ASPP(nn.Module):
    def __init__(self, num_classes):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)

        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(512,
                                    256,
                                    kernel_size=3,
                                    stride=1,
                                    padding=6,
                                    dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(512,
                                    256,
                                    kernel_size=3,
                                    stride=1,
                                    padding=12,
                                    dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(512,
                                    256,
                                    kernel_size=3,
                                    stride=1,
                                    padding=18,
                                    dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)  # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h/16, w/16))
        # (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16.
        # If self.resnet instead is ResNet18_OS8 or ResNet34_OS8,
        # it will be (batch_size, 512, h/8, w/8))

        feature_map_h = feature_map.size()[2]  # (== h/16)
        feature_map_w = feature_map.size()[3]  # (== w/16)

        # print("=====", feature_map.shape)
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))
        # (shape: (batch_size, 256, h/16, w/16))
        # print('-------')

        out_img = self.avg_pool(feature_map)
        # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img,
                             size=(feature_map_h, feature_map_w),
                             mode="bilinear")
        # (shape: (batch_size, 256, h/16, w/16))
        # print('?'*5)
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
        # (shape: (batch_size, 256, h/16, w/16))
        out = self.conv_1x1_4(out)
        # (shape: (batch_size, num_classes, h/16, w/16))

        return out


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


def create_modules(module_defs, img_size, arc):
    # 通过module_defs进行构建模型
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []  # 存储了所有的层，在route、shortcut会使用到。
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()
        '''
        通过type字样不同的类型，来进行模型构建
        '''
        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            size = int(mdef['size'])
            stride = int(mdef['stride']) if 'stride' in mdef else (int(
                mdef['stride_y']), int(mdef['stride_x']))
            pad = (size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module(
                'Conv2d',
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=size,
                    stride=stride,
                    padding=pad,
                    groups=int(mdef['groups']) if 'groups' in mdef else 1,
                    bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d',
                                   nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'leaky':  # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1,
                                                              inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            # 在此处可以添加新的激活函数

        elif mdef['type'] == 'dilatedconv':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            size = int(mdef['size'])
            stride = int(mdef['stride']) if 'stride' in mdef else (int(
                mdef['stride_y']), int(mdef['stride_x']))
            pad = (size + 1) // 2 if int(mdef['pad']) else 0
            modules.add_module(
                'Conv2d',
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=size,
                    stride=stride,
                    padding=pad,
                    groups=int(mdef['groups']) if 'groups' in mdef else 1,
                    dilation=int(mdef['dilation']),
                    bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d',
                                   nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'leaky':  # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1,
                                                              inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            # 在此处可以添加新的激活函数

        elif mdef['type'] == 'maxpool':
            # 最大池化操作
            size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool = nn.MaxPool2d(kernel_size=size,
                                   stride=stride,
                                   padding=int((size - 1) // 2))
            if size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'corner':
            # corner pool
            size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool_1 = nn.MaxPool2d(kernel_size=(size, 1),
                                     stride=stride,
                                     padding=(int((size - 1) // 2),0))
            maxpool_2 = nn.MaxPool2d(kernel_size=(1, size),
                                     stride=stride,
                                     padding=(0,int((size - 1) // 2)))

            if size == 2 and stride == 1:  # yolov3-tiny
                # 这里不考虑yolov3 tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules.add_module('corner_maxpool_1', maxpool_1)
                modules.add_module('corner_maxpool_2', maxpool_2)

        elif mdef['type'] == 'upsample':
            # 通过近邻插值完成上采样
            modules = nn.Upsample(scale_factor=int(mdef['stride']),
                                  mode='nearest')

        elif mdef['type'] == 'se':
            modules.add_module(
                'se_module',
                SELayer(output_filters[-1], reduction=int(mdef['reduction'])))

        elif mdef['type'] == 'cbam':
            ca = ChannelAttention(output_filters[-1], ratio=int(mdef['ratio']))
            sa = SpatialAttention(kernel_size=int(mdef['kernelsize']))
            modules.add_module('channel_attention', ca)
            modules.add_module('spatial attention', sa)

        elif mdef['type'] == 'ppm':
            ppm = PSPModule(output_filters[-1], int(mdef['out']))
            modules.add_module('Pyramid Pooling Module', ppm)

        elif mdef['type'] == 'aspp':
            aspp = ASPP(int(mdef['out']))
            modules.add_module('ASPP', aspp)

        elif mdef['type'] == 'route':
            # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum(
                [output_filters[i + 1 if i > 0 else i] for i in layers])
            # extend表示添加一系列对象
            routs.extend([l if l > 0 else l + i for l in layers])

        elif mdef['type'] == 'shortcut':
            # nn.Sequential() placeholder for 'shortcut' layer
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])

            # modules.add_module(
            #     'stride2conv',
            #     nn.Conv2d(filters,
            #               filters,
            #               kernel_size=1,
            #               stride=2,
            #               bias=False))

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            mask = [int(x) for x in mdef['mask'].split(',')]  # anchor mask
            modules = YOLOLayer(
                anchors=mdef['anchors'][mask],  # anchor list
                nc=int(mdef['classes']),  # number of classes
                img_size=img_size,  # (416, 416)
                yolo_index=yolo_index,  # 0, 1 or 2
                arc=arc)  # yolo architecture

            # 这是在focal loss文章中提到的为卷积层添加bias
            # 主要用于解决样本不平衡问题
            # (论文地址 https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            # pw 代表pretrained weights
            try:
                if arc == 'defaultpw' or arc == 'Fdefaultpw':
                    # default with positive weights
                    b = [-5.0, -5.0]  # obj, cls
                elif arc == 'default':
                    # default no pw (40 cls, 80 obj)
                    b = [-5.0, -5.0]
                elif arc == 'uBCE':
                    # unified BCE (80 classes)
                    b = [0, -9.0]
                elif arc == 'uCE':
                    # unified CE (1 background + 80 classes)
                    b = [10, -0.1]
                elif arc == 'Fdefault':
                    # Focal default no pw (28 cls, 21 obj, no pw)
                    b = [-2.1, -1.8]
                elif arc == 'uFBCE' or arc == 'uFBCEpw':
                    # unified FocalBCE (5120 obj, 80 classes)
                    b = [0, -6.5]
                elif arc == 'uFCE':
                    # unified FocalCE (64 cls, 1 background + 80 classes)
                    b = [7.7, -1.1]

                # len(mask) = 3
                bias = module_list[-1][0].bias.view(len(mask), -1)

                # 255 to 3x85
                bias[:, 4] += b[0] - bias[:, 4].mean()  # obj
                bias[:, 5:] += b[1] - bias[:, 5:].mean()  # cls

                # list of tensors [3x85, 3x85, 3x85]
                module_list[-1][0].bias = torch.nn.Parameter(bias.view(-1))

            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # 将module内容保存在module_list中。
        module_list.append(modules)
        # 保存所有的filter个数
        output_filters.append(filters)

    return module_list, routs


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


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, arc):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # 该YOLOLayer分配给每个grid的anchor的个数
        self.nc = nc  # 类别个数
        self.no = nc + 5  # 每个格子对应输出的维度 class + 5 中5代表x,y,w,h,conf
        self.nx = 0  # 初始化x方向上的格子数量
        self.ny = 0  # 初始化y方向上的格子数量
        self.arc = arc

        if ONNX_EXPORT:  # grids must be computed in __init__
            stride = [32, 16, 8][yolo_index]  # stride of this layer
            nx = int(img_size[1] / stride)  # number x grid points
            ny = int(img_size[0] / stride)  # number y grid points
            create_grids(self, img_size, (nx, ny))

    def forward(self, p, img_size, var=None):
        '''
        onnx代表开放式神经网络交换
        pytorch中的模型都可以导出或转换为标准ONNX格式
        在模型采用ONNX格式后，即可在各种平台和设备上运行
        在这里ONNX代表规范化的推理过程
        '''
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)
        # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny,
                   self.nx).permute(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return p

        elif ONNX_EXPORT:
            m = self.na * self.nx * self.ny
            grid_xy = self.grid_xy.repeat((1, self.na, 1, 1, 1)).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(
                (1, 1, self.nx, self.ny, 1)).view(m, 2) / self.ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid_xy  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(
                p[:, 4:5])  # conf
            return p_cls, xy / self.ng, wh

        else:  # 测试推理过程
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(
                io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride

            if 'default' in self.arc:  # seperate obj and cls
                torch.sigmoid_(io[..., 4])
            elif 'BCE' in self.arc:  # unified BCE (80 classes)
                torch.sigmoid_(io[..., 5:])
                io[..., 4] = 1
            elif 'CE' in self.arc:  # unified CE (1 background + 80 classes)
                io[..., 4:] = F.softmax(io[..., 4:], dim=4)
                io[..., 4] = 1

            if self.nc == 1:
                io[..., 5] = 1
                # single-class model https://github.com/ultralytics/yolov3/issues/235

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, self.no), p


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), arc='default'):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs,
                                                      img_size, arc)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)
        # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)
        # (int64) number of images seen during training
        self.show = True

    def forward(self, x, var=None):
        img_size = x.shape[-2:]
        layer_outputs = []
        output = []

        for i, (mdef,
                module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']

            # if self.show:
            #     print("第%3d层: %13s | " % (i, mtype), "shape:", x.shape)

            if mtype in [
                    'convolutional', 'upsample', 'maxpool', 'se',
                    'dilatedconv', 'ppm', 'aspp'
            ]:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    try:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        layer_outputs[layers[1]] = F.interpolate(
                            layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layers], 1)

            elif mtype == 'shortcut':
                # x = module(x)
                x = x + layer_outputs[int(mdef['from'])]

            elif mtype == 'corner':
                x1 = module[0](x)
                # print("after x1:", x1.shape)
                x2 = module[1](x)

                # print("after x2:",x2.shape)   
                x = x1 + x2

            elif mtype == 'cbam':
                ca = module[0]
                sa = module[1]
                x = ca(x) * x
                x = sa(x) * x
            elif mtype == 'yolo':
                output.append(module(x, img_size))
            layer_outputs.append(x if i in self.routs else [])

        self.show = False

        if self.training:
            return output
        elif ONNX_EXPORT:
            x = [torch.cat(x, 0) for x in zip(*output)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        # model_info(self)  # yolov3-spp reduced from 225 to 152 layers


def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs)
            if x['type'] == 'yolo']  # [82, 94, 106] for yolov3


def create_grids(self,
                 img_size=416,
                 ng=(13, 13),
                 device='cpu',
                 type=torch.float32):
    nx, ny = ng  # 网格尺寸
    self.img_size = max(img_size)
    #下采样倍数为32
    self.stride = self.img_size / max(ng)

    # 划分网格，构建相对左上角的偏移量
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view(
        (1, 1, ny, nx, 2))

    # 处理anchor，将其除以下采样倍数
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1,
                                          2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(
            f, dtype=np.int32,
            count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(
            f, dtype=np.int64,
            count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(
            zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv_layer = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                    bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                    bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                    bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                    bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                    conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(
                conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

    return cutoff


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(
                zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {
            'epoch': -1,
            'best_fitness': None,
            'training_results': None,
            'model': model.state_dict(),
            'optimizer': None
        }

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if weights and not os.path.isfile(weights):
        d = {
            'yolov3-spp.weights': '16lYS4bcIdM2HdmyJBVDOvt3Trx6N3W2R',
            'yolov3.weights': '1uTlyDWlnaqXcsKOktP5aH_zRDbfcDp-y',
            'yolov3-tiny.weights': '1CCF-iNIIkYesIDzaPvdwlcf7H9zSsKZQ',
            'yolov3-spp.pt': '1f6Ovy3BSq2wYq4UfvFUpxJFNDFfrIDcR',
            'yolov3.pt': '1SHNFyoe5Ni8DajDNEqgB2oVKBb_NoEad',
            'yolov3-tiny.pt': '10m_3MlpQwRtZetQxtksm9jqHrPTHZ6vo',
            'darknet53.conv.74': '1WUVBid-XuoUBmvzBVUCBl_ELrzqwA8dJ',
            'yolov3-tiny.conv.15': '1Bw0kCpplxUqyRYAJr9RY9SGnOJbo9nEj',
            'ultralytics49.pt': '158g62Vs14E3aj7oPVPuEnNZMKFNgGyNq',
            'ultralytics68.pt': '1Jm8kqnMdMGUUxGo8zMFZMJ0eaPwLkxSG'
        }

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:  # download from pjreddie.com
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        # Error check
        if not (r == 0 and os.path.exists(weights)
                and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)
