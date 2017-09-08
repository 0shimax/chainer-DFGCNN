from chainer import initializers
from chainer import link
from chainer import variable
# from chainer.utils import argument

import filter_dict_conv2d_func


class FilterDictConv2d(link.Link):
    def __init__(self, in_channels, out_channels, ksize=None,
                 initialW=None, initial_bias=None, **kwargs):
        super().__init__()

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.ksize = ksize  # w*h
        self.out_channels = out_channels

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            # maybe don't need
            if in_channels is not None:
                self._initialize_params(in_channels, ksize)

    def _initialize_params(self, in_channels, ksize):
        W_shape = (in_channels, self.out_channels, ksize)
        self.W.initialize(W_shape)

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return filter_dict_conv2d_func.filter_dict_conv2d(x, self.W)
