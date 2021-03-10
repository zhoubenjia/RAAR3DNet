"""RAAR3DNet architecture.
 The model is introduced in:
     Regional Attention with Architecture-Rebuilt 3D Network for RGB-D Gesture Recognition
     Benjia Zhou, Yunan Li, Jun Wan
     https://arxiv.org/pdf/2102.05348.pdf
 """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
from collections import OrderedDict
import pdb
from .Operations import *
from .Genotypes import PRIMITIVES_INCEPTION
from .Genotypes import Genotype

PRIMITIVES = PRIMITIVES_INCEPTION



class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
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
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
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
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


def channel_shuffle(x, groups):
    batchsize, num_channels, length, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    # ([2, 32, 4, 28, 28])--> ([2, 4, 8, 4, 28, 28])
    x = x.view(batchsize, groups, channels_per_group, length, height, width)
    # -->([2, 8, 4, 4, 28, 28])
    x = torch.transpose(x, 1, 2).contiguous()  # not change x dim
    # flatten
    x = x.view(batchsize, -1, length, height, width)
    return x
class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._k = 2
        for primitive in PRIMITIVES:
            op = OPS[primitive](C // self._k, stride, False)
            self._ops.append(op)

    def forward(self, x, weights):
        # channel proportion k=4
        dim_2 = x.shape[1]
        xtemp = x[:, :  dim_2 // self._k, :, :].cuda()  # [2, 8, 4, 28, 28]
        xtemp2 = x[:, dim_2 // self._k:, :, :].cuda()  # [2, 24, 4, 28, 28]
        temp1 = sum(w.cuda() * op(xtemp).cuda() for w, op in zip(weights, self._ops))

        ans = torch.cat([temp1, xtemp2], dim=1)

        ans = channel_shuffle(ans.cuda(), self._k)
        # ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        # except channe shuffle, channel shift also works
        return ans


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction_prev):
        super(Cell, self).__init__()
        # cell的输出为各层输出的拼接，维度很高，所以在输入加了两个预处理preprocess0和preprocess1。
        if reduction_prev:
            self.preprocess0 = VaniConv3d(C_prev_prev, C, 1, [1, 2, 2], 0, affine=False)
        else:
            self.preprocess0 = VaniConv3d(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = VaniConv3d(C_prev, C, 1, 1, 0, affine=False)

        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights, weights2):

        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            # weights[offset + j]=([0.1429, 0.1426, 0.1429, 0.1428, 0.1429, 0.1429, 0.1429], weights2[offset + j]=0.49999
            s_temp = sum(weights2[offset + j].cuda() * self._ops[offset + j](h, weights[offset + j]).cuda() for j, h in
                         enumerate(states))
            offset += len(states)
            states.append(s_temp)

        return torch.cat(states[-self._multiplier:], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
  The model is introduced in:
      Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
      Joao Carreira, Andrew Zisserman
      https://arxiv.org/pdf/1705.07750v1.pdf.
  See also the Inception architecture, introduced in:
      Going deeper with convolutions
      Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
      Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
      http://arxiv.org/pdf/1409.4842v1.pdf.
  """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
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

    def __init__(self, num_classes, criterion, local_rank, steps=4, multiplier=4, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
    Args:
      num_classes: The number of outputs in the logit layer (default 400, which
          matches the Kinetics dataset).
      spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
          before returning (default True).
      final_endpoint: The model contains many possible endpoints.
          `final_endpoint` specifies the last endpoint for the model to be built
          up to. In addition to the output at `final_endpoint`, all the outputs
          at endpoints up to `final_endpoint` will also be returned, in a
          dictionary. `final_endpoint` must be one of
          InceptionI3d.VALID_ENDPOINTS (default 'Logits').
      name: A string (optional). The name of this module.
    Raises:
      ValueError: if `final_endpoint` is not recognized.
    """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

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
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'cell1'
        self.end_points[end_point] = nn.ModuleList()
        C_curr = 64
        C_prev_prev, C_prev, C_curr = C_curr * 3, C_curr * 3, C_curr
        reduction_prev = False
        for i in range(2):
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction_prev)
            self.end_points[end_point] += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
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
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction_prev)
            self.end_points[end_point] += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'cell3'
        self.end_points[end_point] = nn.ModuleList()
        C_prev_prev, C_prev, C_curr = C_prev, C_prev, C_curr
        reduction_prev = False
        for i in range(2):
            if i == 1:
                C_curr *= 2
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction_prev)
            self.end_points[end_point] += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        if self._final_endpoint == end_point: return

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
        self._initialize_alphas(local_rank)

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
                        weights = F.softmax(self.alphas_normal1, dim=-1)  # 14x7(edge x operations)
                        n = 3
                        start = 2
                        weights2 = F.softmax(self.betas_normal1[0:2], dim=-1)  # self.betas_normal16: 14x1
                        for i in range(self._steps - 1):
                            end = start + n
                            tw2 = F.softmax(self.betas_normal1[start:end], dim=-1)  # 2-5, 5-9, 9-14
                            start = end
                            n += 1
                            weights2 = torch.cat([weights2, tw2], dim=0)

                        s0, s1 = s1, cell(s0, s1, weights, weights2)
                    x = s1

                elif end_point == 'cell2':
                    s0 = s1 = x
                    for ii, cell in enumerate(self._modules[end_point]):
                        weights = F.softmax(self.alphas_normal2, dim=-1)  # 14x7(edge x operations)
                        n = 3
                        start = 2
                        weights2 = F.softmax(self.betas_normal2[0:2], dim=-1)  # self.betas_normal16: 14x1
                        for i in range(self._steps - 1):
                            end = start + n
                            tw2 = F.softmax(self.betas_normal2[start:end], dim=-1)  # 2-5, 5-9, 9-14
                            start = end
                            n += 1
                            weights2 = torch.cat([weights2, tw2], dim=0)
                        s0, s1 = s1, cell(s0, s1, weights, weights2)
                    x = s1
                elif end_point == 'cell3':
                    s0 = s1 = x
                    for ii, cell in enumerate(self._modules[end_point]):
                        weights = F.softmax(self.alphas_normal3, dim=-1)  # 14x7(edge x operations)
                        n = 3
                        start = 2
                        weights2 = F.softmax(self.betas_normal3[0:2], dim=-1)  # self.betas_normal16: 14x1
                        for i in range(self._steps - 1):
                            end = start + n
                            tw3 = F.softmax(self.betas_normal3[start:end], dim=-1)  # 2-5, 5-9, 9-14
                            start = end
                            n += 1
                            weights2 = torch.cat([weights2, tw3], dim=0)
                        s0, s1 = s1, cell(s0, s1, weights, weights2)

                    x = s1
                else:
                    x = self._modules[end_point](x)  # use _modules to work with dataparallel

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        return logits.squeeze()

    def _loss(self, inputs, target):
        logits = self(inputs)
        return self._criterion(logits, target)

    def _initialize_alphas(self, local_rank):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal1 = Variable(1e-3 * torch.randn(k, num_ops).cuda(local_rank), requires_grad=True)  # net normal
        self.betas_normal1 = Variable(1e-3 * torch.randn(k).cuda(local_rank), requires_grad=True)  # edge normal
        self.alphas_normal2 = Variable(1e-3 * torch.randn(k, num_ops).cuda(local_rank), requires_grad=True)  # net normal
        self.betas_normal2 = Variable(1e-3 * torch.randn(k).cuda(local_rank), requires_grad=True)  # edge normal
        self.alphas_normal3 = Variable(1e-3 * torch.randn(k, num_ops).cuda(local_rank), requires_grad=True)  # net normal
        self.betas_normal3 = Variable(1e-3 * torch.randn(k).cuda(local_rank), requires_grad=True)  # edge normal
        self._arch_parameters = [
            self.alphas_normal1,
            self.betas_normal1,
            self.alphas_normal2,
            self.betas_normal2,
            self.alphas_normal3,
            self.betas_normal3
        ]
    def resume_arch_parameters(self, parames, local_rank):
        self.alphas_normal1 = Variable(parames[0].cuda(local_rank), requires_grad=True)  # net normal
        self.betas_normal1 = Variable(parames[1].cuda(local_rank), requires_grad=True)  # edge normal
        self.alphas_normal2 = Variable(parames[2].cuda(local_rank), requires_grad=True)  # net normal
        self.betas_normal2 = Variable(parames[3].cuda(local_rank), requires_grad=True)  # edge normal
        self.alphas_normal3 = Variable(parames[4].cuda(local_rank), requires_grad=True)  # net normal
        self.betas_normal3 = Variable(parames[5].cuda(local_rank), requires_grad=True)  # edge normal

    def arch_parameters(self):
        return self._arch_parameters

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)

    def genotype(self):
        def _parse(weights, weights2):
            gene = []
            n = 2
            start = 0
            # 对于每层，由归一化后的参数乘积排序，取最大的两个非空操作为边。每条边再确定最佳操作。
            for i in range(self._steps):
                end = start + n
                # W: [[0.14320895 0.14271285 0.14264019 0.14287315 0.14279652 0.14292398, 0.14284438]
                #   [0.1430605  0.14276284 0.14267652 0.14286381 0.1430042  0.14296356, 0.14266858]]
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()  # [0.4998488 0.5001512]

                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]  # each operation * weights2

                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                # edges=1, 0

                # edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        # start = 2跳过两个输入节点, 循环处理3个中间节点, 统一将同层betas_normal16送 Softmax 归一化
        n = 3
        start = 2
        weightsn1 = F.softmax(self.betas_normal1[0:2], dim=-1)
        weightsn2 = F.softmax(self.betas_normal2[0:2], dim=-1)
        weightsn3 = F.softmax(self.betas_normal3[0:2], dim=-1)

        for i in range(self._steps - 1):
            end = start + n
            # print(self.betas_reduce[start:end])
            tn1 = F.softmax(self.betas_normal1[start:end], dim=-1)
            tn2 = F.softmax(self.betas_normal2[start:end], dim=-1)
            tn3 = F.softmax(self.betas_normal3[start:end], dim=-1)
            start = end
            n += 1
            weightsn1 = torch.cat([weightsn1, tn1], dim=0)
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
            weightsn3 = torch.cat([weightsn3, tn3], dim=0)

        gene_normal1 = _parse(F.softmax(self.alphas_normal1, dim=-1).data.cpu().numpy(),
                              weightsn1.data.cpu().numpy())
        gene_normal2 = _parse(F.softmax(self.alphas_normal2, dim=-1).data.cpu().numpy(),
                              weightsn2.data.cpu().numpy())
        gene_normal3 = _parse(F.softmax(self.alphas_normal3, dim=-1).data.cpu().numpy(),
                              weightsn3.data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal1=gene_normal1, normal_concat1=concat,
            normal2=gene_normal2, normal_concat2=concat,
            normal3=gene_normal3, normal_concat3=concat,
        )
        return genotype


if __name__ == '__main__':
    import os, torch

    sample_size = 224
    sample_duration = 32
    num_classes = 40
    resnet_shortcut = 'A'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    model = InceptionI3d(num_classes, torch.nn.CrossEntropyLoss(), in_channels=3).cuda()

    inputs = torch.randn(2, 3, 64, 112, 112).cuda()  # shape (C x T x H x W)
    outputs = model(inputs)
    print(outputs.shape)
