import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer import function
from chainer.utils import argument
from chainer.utils import conv
from chainer.utils import type_check


class FilterDictConv2dFunction(function.Function):
    def __init__(self):
        pass

    def forward(self, inputs):
        '''
        in: N_i*D
        out: N_i*E*M
        '''
        self.xp = cuda.get_array_module(*inputs)
        x, W = inputs

        return self.xp.tensordot(x, W, axes=([2],[0])),  # (N_i, E, M)

    def backward(self, inputs, grad_outputs):
        # x: N_i*D
        # W: D*E*M
        x, W = inputs
        gy, = grad_outputs  # N_i*E*M
        gx = self.xp.tensordot(gy, W.T, axes=([3,2], [0,1]))
        gW = self.xp.tensordot(gy.T, x, axes=([3,2], [0,1])).T
        return gx, gW


def filter_dict_conv2d(x, W):
    func = FilterDictConv2dFunction()
    return func(x, W)
