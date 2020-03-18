import os

import numpy as np


def parse_model_cfg(path):
    # path参数为: cfg/yolov3-tiny.cfg
    if not path.endswith('.cfg'):
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):
        path = 'cfg' + os.sep + path

    with open(path, 'r') as f:
        lines = f.read().split('\n')

    # 去除以#开头的，属于注释部分的内容
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    mdefs = []  # 模块的定义
    for line in lines:
        if line.startswith('['):  # 标志着一个模块的开始
            '''
            eg:
            [shortcut]
            from=-3
            activation=linear
            '''
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1][
                    'batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if 'anchors' in key:
                mdefs[-1][key] = np.array([float(x)
                                           for x in val.split(',')]).reshape(
                                               (-1, 2))  # np anchors
            else:
                mdefs[-1][key] = val.strip()

    # Check all fields are supported
    supported = [
        'type', 'batch_normalize', 'filters', 'size', 'stride', 'pad',
        'activation', 'layers', 'groups', 'from', 'mask', 'anchors', 'classes',
        'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random', 'stride_x',
        'stride_y', 'ratio', 'reduction', 'kernelsize', 'dilation', 'out', 'branch', 'scale'
    ]

    f = []  # fields
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not any(
        u
    ), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (
        u, path)

    return mdefs


def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists(
            'data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options
