# encoding: utf-8


# import torch
# from torch import nn
from .batch_norm import get_norm
from mindspore import nn, ops
import mindspore


# class Non_local(nn.Module):
class Non_local(nn.Cell):
    def __init__(self, in_channels, bn_norm, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.SequentialCell(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            get_norm(bn_norm, self.in_channels),
        )
        # self.W = nn.Sequential(
        #     nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
        #               kernel_size=1, stride=1, padding=0),
        #     get_norm(bn_norm, self.in_channels),
        # )

        # nn.init.constant_(self.W[1].weight, 0.0)
        self.W[1].gamma.set_data(mindspore.common.initializer.initializer("zeros", self.W[1].gamma.shape, self.W[1].gamma.dtype))
        # nn.init.constant_(self.W[1].bias, 0.0)
        self.W[1].beta.set_data(mindspore.common.initializer.initializer("zeros", self.W[1].beta.shape, self.W[1].beta.dtype))

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    # def forward(self, x):
    def construct(self, x):
        """
                :param x: (b, t, h, w)
                :return x: (b, t, h, w)
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = ops.Transpose()(mindspore.Tensor(g_x), (0, 2, 1))
        # g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = ops.Transpose()(mindspore.Tensor(theta_x), (0, 2, 1))
        # theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = ops.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = ops.matmul(f_div_C, g_x)
        y = ops.Transpose()(mindspore.Tensor(y), (0, 2, 1))
        # y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z
