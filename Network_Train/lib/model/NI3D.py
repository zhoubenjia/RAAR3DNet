"""RAAR3DNet architecture.
 The model is introduced in:
     Regional Attention with Architecture-Rebuilt 3D Network for RGB-D Gesture Recognition
     Benjia Zhou, Yunan Li, Jun Wan
     https://arxiv.org/abs/2102.05348v2
 """

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Operations import *
from torch.autograd import Variable
import pdb
from utils import load_pretrained_checkpoint
import logging
class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction_prev, normal):
        super(Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = VaniConv3d(C_prev_prev, C, 1, [1, 2, 2], 0)
        else:
            self.preprocess0 = VaniConv3d(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = VaniConv3d(C_prev, C, 1, 1, 0)

        # cell1
        if normal == 1:
            op_names, indices = zip(*genotype.normal1)
            concat = genotype.normal_concat1
        # cell2
        if normal == 2:
            op_names, indices = zip(*genotype.normal2)
            concat = genotype.normal_concat2
        # cell3
        if normal == 3:
            op_names, indices = zip(*genotype.normal3)
            concat = genotype.normal_concat3

        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        # pdb.set_trace()

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)
class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)


    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

class InceptionI3d(nn.Module):
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'cell1',
        'MaxPool3d_4a_3x3',
        'cell2',
        'MaxPool3d_5a_2x2',
        'cell3',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes, genotype, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()

        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.MaxpoolSpa = nn.MaxPool3d(kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=[1, 2, 2])

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1, name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'cell1'
        self.end_points[end_point] = nn.ModuleList()
        C_curr = 64
        C_prev_prev, C_prev, C_curr = C_curr * 3, C_curr * 3, C_curr
        reduction_prev = False
        for i in range(2):
            cell = Cell(genotype, C_prev_prev=C_prev_prev, C_prev=C_prev, C=C_curr, reduction_prev=reduction_prev, normal=1)
            self.end_points[end_point] += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'cell2'
        self.end_points[end_point] = nn.ModuleList()
        C_prev_prev, C_prev, C_curr = C_prev, C_prev, C_curr
        for i in range(5):
            if i == 2:
                C_curr *= 2
                reduction_prev = False
            else:
                reduction_prev = False
            cell = Cell(genotype, C_prev_prev=C_prev_prev, C_prev=C_prev, C=C_curr, reduction_prev=reduction_prev, normal=2)
            self.end_points[end_point] += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        # self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return
        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'cell3'
        self.end_points[end_point] = nn.ModuleList()
        C_prev_prev, C_prev, C_curr = C_prev, C_prev, C_curr
        reduction_prev = False
        for i in range(2):
            if i == 1:
                C_curr *= 2
            cell = Cell(genotype, C_prev_prev=C_prev_prev, C_prev=C_prev, C=C_curr, reduction_prev=reduction_prev, normal=3)
            self.end_points[end_point] += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        end_point = 'Logits'
        # self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
        #                             stride=(1, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                if end_point == 'cell1':
                    s0 = s1 = x
                    for ii, cell in enumerate(self._modules[end_point]):
                        s0, s1 = s1, cell(s0, s1)
                    x = s1
                    f1 = x
                elif end_point == 'cell2':
                    s0 = s1 = x
                    for ii, cell in enumerate(self._modules[end_point]):
                        s0, s1 = s1, cell(s0, s1)
                    x = s1
                    f2 = x
                elif end_point == 'cell3':
                    s0 = s1 = x
                    for ii, cell in enumerate(self._modules[end_point]):
                        s0, s1 = s1, cell(s0, s1)
                    x = s1
                else:
                    x = self._modules[end_point](x)  # use _modules to work with dataparallel

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze()
        # logits is batch X time X classes, which is what we want to work with
        return logits, (f1, f2)

def Network(num_classes, genotype, local_rank, pretrained=None):
    Net = InceptionI3d(num_classes=num_classes, genotype=genotype)
    if pretrained:
        Net.replace_logits(27)
        checkpoint = pretrained
        Net = load_pretrained_checkpoint(Net, checkpoint, local_rank)
        Net.replace_logits(num_classes)
        logging.info("Load Pre-trained model state_dict Done !")
    return Net

if __name__ == '__main__':
    import torch
    from collections import namedtuple
    Genotype = namedtuple('Genotype', 'normal1 normal_concat1 normal2 normal_concat2 normal3 normal_concat3')
    genotype = Genotype(
        normal1=[('conv_1x1x1', 0), ('conv_3x3x3', 1), ('Max_pool_3x3x3', 0), ('skip_connect', 1), ('conv_1x1x1', 2),
                 ('Max_pool_3x3x3', 0), ('skip_connect', 0), ('conv_3x3x3', 2)], normal_concat1=range(2, 6),
        normal2=[('Max_pool_3x3x3', 0), ('conv_3x3x3', 1), ('conv_1x1x1', 2), ('conv_3x3x3', 0), ('skip_connect', 3),
                 ('conv_1x1x1', 2), ('skip_connect', 3), ('skip_connect', 2)], normal_concat2=range(2, 6),
        normal3=[('conv_3x3x3', 1), ('conv_3x3x3', 0), ('conv_3x3x3', 1), ('conv_1x1x1', 2), ('conv_1x1x1', 3),
                 ('conv_1x1x1', 2), ('conv_3x3x3', 1), ('conv_1x1x1', 4)], normal_concat3=range(2, 6))

    model = Network(249, genotype)
    inputs = torch.randn(2, 3, 32, 224, 224)
    output = model(inputs)
    print(output.shape)
