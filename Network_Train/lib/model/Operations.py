import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity(),
    'conv_1x1x1': lambda C, stride, affine: conv3d(C, C, [1, 1, 1], stride=[1, 1, 1], padding=0, use_bias=False, affine=affine),
    'conv_3x3x3': lambda C, stride, affine: conv3d(C, C, [3, 3, 3],  stride=[1, 1, 1], padding=0, use_bias=False, affine=affine),
    'Max_pool_3x3x3': lambda C, stride, affine: MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0),
    'Vaniconv_3x3x3': lambda C, stride, affine: VaniConv3d(C, C, 3, stride, 1, affine=affine),
    'dil_3x3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, padding=2, dilation=2, affine=affine),
    'conv_1x3x3': lambda C, stride, affine: VaniConv3d_Spatial_1x3x3(C, C, 3, stride, 1, affine=affine),
    'conv_3x1x1': lambda C, stride, affine: VaniConv3d_Temporal_3x1x1(C, C, 3, stride, 1, affine=affine),
}

class Conv3d_5x5x5(nn.Module):
    def __init__(self, C_in, C_out, affine, conv_name):
        super(Conv3d_5x5x5, self).__init__()

        self.conv_1x1x1 = conv3d(C_in, C_out, [1, 1, 1], [1, 1, 1], 0, False)
        if conv_name == 'conv':
            self.conv_3x3x3 = conv3d(C_in, C_out, [3, 3, 3], [1, 1, 1], 0, False)
        elif conv_name == 'dil':
            self.conv_3x3x3 = DilConv(C_in, C_out, 3, 1, padding=2, dilation=2, affine=affine)
        elif conv_name == 'vani':
            self.conv_3x3x3 = VaniConv3d(C_in, C_out, 3, 1, 1, affine=affine)

    def forward(self, x):
        return self.conv_3x3x3(self.conv_1x1x1(x))

class MaxPool_3x3x3(nn.Module):
    def __init__(self, C_in, C_out):
        super(MaxPool_3x3x3, self).__init__()
        self.conv_1x1x1 = conv3d(C_in, C_out, [1, 1, 1], [1, 1, 1], 0, False)
        self.MaxPool_3x3x3 = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)

    def forward(self, x):
        return self.conv_1x1x1(self.MaxPool_3x3x3(x))


class conv3d(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, use_bias, affine=True, activation_fn=F.relu):
        super(conv3d, self).__init__()
        self._kernel_shape = kernel_size
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.padding = padding
        self.op = nn.Sequential(
            nn.Conv3d(in_channels=C_in,  out_channels=C_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=use_bias, groups=C_in),
            nn.BatchNorm3d(C_out, eps=0.001, momentum=0.01, affine=affine), # affine:一个布尔值，当设为true，给该层添加可学习的仿射变换参数
        )
    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        x = self.op(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

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

class VaniConv3d(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(VaniConv3d, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class VaniConv3d_Spatial_1x3x3(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(VaniConv3d_Spatial_1x3x3, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_out, [1, 3, 3], stride=stride, padding=[0, 1, 1], bias=False),
            nn.BatchNorm3d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class VaniConv3d_Temporal_3x1x1(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(VaniConv3d_Temporal_3x1x1, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_out, [3, 1, 1], stride=stride, padding=[1, 0, 0], bias=False),
            nn.BatchNorm3d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, # dilation=2的3*3卷积，感受野是7*7.
                      bias=False),
            nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv3d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce_Spatial(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce_Spatial, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv3d(C_in, C_out // 2, 1, stride=[1, 2, 2], padding=0, bias=False)
        self.conv_2 = nn.Conv3d(C_in, C_out // 2, 1, stride=[1, 2, 2], padding=0, bias=False)
        self.bn = nn.BatchNorm3d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        # out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class FactorizedReduce_SpatialTemporal(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce_SpatialTemporal, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm3d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        # out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class MaxPool_ST(nn.Module):

    def __init__(self, stride):
        super(MaxPool_ST, self).__init__()
        self.Maxpool = nn.MaxPool3d(stride, stride)

    def forward(self, x):
        x = self.Maxpool(x)

        return x

