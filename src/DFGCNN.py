import chainer
import chainer.functions as F
import chainer.links as L
from chainer import config
import itertools

from filter_dict_conv2d import FilterDictConv2d


class DFGCNN(chainer.Chain):
    def __init__(self, in_channels, n_class, ksize):
        super().__init__()
        with self.init_scope():
            self.dfg = FilterDictConv2d(in_channels, n_class, ksize)  # out:N_i*E*M
            self.weight_target_node = \
                L.DepthwiseConvolution2D(in_channels, ksize, 1)  # out: D*M
            # self.out_adjacent_node = \
            #     L.ConvolutionND(ksize, in_channels, in_channels, 1)
        self.in_channels = in_channels
        self.n_class = n_class
        self.img_h, self.img_w = 28, 28

        adjacent_matrix = {}
        for r, c in itertools.product(range(28), range(28)):
            if r==0 and c==0:
                adjacent_matrix[r*self.img_w+c] = [r*self.img_w+c+1, (r+1)*self.img_w+c+1, (r+1)*self.img_w+c]
            elif r==0 and c==27:
                adjacent_matrix[r*self.img_w+c] = [r*self.img_w+c-1, (r+1)**self.img_w+c-1, (r+1)*self.img_w+c]
            elif r==0 and 0<c<27:
                adjacent_matrix[r*self.img_w+c] = [
                    r*self.img_w+c-1, r*self.img_w+c+1,
                    (r+1)*self.img_w+c-1, (r+1)*self.img_w+c, (r+1)*self.img_w+c+1]
            elif r==27 and c==0:
                adjacent_matrix[r*self.img_w+c] = [r*self.img_w+c+1, (r-1)*self.img_w+c+1, (r-1)*self.img_w+c]
            elif r==27 and c==27:
                adjacent_matrix[r*self.img_w+c] = [r*self.img_w+c-1, (r-1)*self.img_w+c-1, (r-1)*self.img_w+c]
            elif r==27 and 0<c<27:
                adjacent_matrix[r*self.img_w+c] = [
                    r*self.img_w+c-1, r*self.img_w+c+1,
                    (r-1)*self.img_w+c-1, (r-1)*self.img_w+c, (r-1)*self.img_w+c+1]
            elif 0<r<27 and c==0:
                adjacent_matrix[r*self.img_w+c] = [
                    (r-1)*self.img_w+c, (r-1)*self.img_w+c+1,
                    r*self.img_w+c+1, (r+1)*self.img_w+c, (r+1)*self.img_w+c+1]
            elif 0<r<27 and c==27:
                adjacent_matrix[r*self.img_w+c] = [
                    (r-1)*self.img_w+c, (r-1)*self.img_w+c-1,
                    r*self.img_w+c-1, (r+1)*self.img_w+c, (r+1)*self.img_w+c-1]
            elif 0<r<27 and 0<c<27:
                adjacent_matrix[r*self.img_w+c] = [
                    (r-1)*self.img_w+c-1, (r-1)*self.img_w+c, (r-1)*self.img_w+c+1,
                    r*self.img_w+c-1, r*self.img_w+c+1,
                    (r+1)*self.img_w+c-1, (r+1)*self.img_w+c, (r+1)*self.img_w+c+1]

            self.adjacent_matrix = adjacent_matrix

    def predict(self, x, node_idx):
        bs, n_node  = x.shape[:2]
        in_ch = x.shape[-1] if len(x.shape[:2])>=3 else 1
        x = F.reshape(x, (-1, in_ch, 28*28, 1))

        r = node_idx//self.img_w
        c = node_idx - r*self.img_h

        target_node_info = [node_idx] + self.adjacent_matrix[node_idx]
        n_adjacent = len(target_node_info)

        # target_nodes_features = F.get_item(x, [range(bs), target_node_info, range(in_ch)])
        target_nodes_features = F.get_item(x, [slice(0,bs), slice(0,in_ch), target_node_info, None])
        target_nodes_features = F.reshape(target_nodes_features, (-1, in_ch, n_adjacent, 1))

        edge_weights_base = self.weight_target_node(target_nodes_features)  # bs, D*M, N_i, 1
        # reshape to (bs, D, M, N_i)
        edge_weights_base = F.reshape(edge_weights_base, (-1, in_ch, config.M, n_adjacent))
        # transpose to (M, bs, N_i, D)
        edge_weights_base = F.transpose(edge_weights_base, (2, 0, 3, 1))
        # reshape to (bs, N_i, D)
        target_nodes_features = F.reshape(target_nodes_features, (bs, n_adjacent, in_ch))
        _target_nodes_features = F.broadcast_to(target_nodes_features, (config.M, bs, n_adjacent, in_ch))
        # merginal for D
        edge_weights_base = F.sum(edge_weights_base*_target_nodes_features, axis=3)

        bc_edge_w = F.broadcast_to(F.sum(edge_weights_base, axis=0), (config.M, bs, n_adjacent))
        edge_weights = F.where(bc_edge_w.data==0, chainer.Variable(self.xp.zeros_like(bc_edge_w.data)), edge_weights_base/bc_edge_w)

        # edge_weights = edge_weights_base \
        #                 /F.broadcast_to(F.sum(edge_weights_base, axis=0), (config.M, bs, n_adjacent)) \
        #                 /n_adjacent  # normalize for number of nodes

        # normalize
        # out: # (M, bs, N_i)
        edge_weights /= n_adjacent
        edge_weights = F.transpose(edge_weights, (1, 2, 0))  # (bs, N_i, M)

        out = self.dfg(target_nodes_features)  # (bs, N_i, E, M)
        out = F.transpose(out, (2, 0, 1, 3))  # (E, bs, N_i, M)
        # weighted
        weighted_out = F.broadcast_to(edge_weights, (self.n_class, bs, n_adjacent, config.M))*out
        results = F.transpose(F.sum(weighted_out, axis=(2, 3)), (1,0))
        return results

    def __call__(self, x, t):
        self.n_node = len(self.adjacent_matrix)  # include node itself

        # compute node_idx==0
        y = self.predict(x, 0)

        for node_idx in range(1, self.n_node):
            y += self.predict(x, node_idx)
            # self.loss += F.softmax_cross_entropy(y, t[:, node_idx])
            # self.accuracy += F.accuracy(y, t[:,node_idx])
        y /= self.n_node
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        # self.loss /= self.n_adjacent
        # self.accuracy /= self.n_adjacent
        chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
        return self.loss
