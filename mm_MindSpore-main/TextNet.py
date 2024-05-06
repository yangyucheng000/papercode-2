# import torch
# from torch import nn
from torch.nn import functional as F
import mindspore
class TextNet(mindspore.nn.Cell):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=2):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TextNet, self).__init__()
        self.module_name = "txt_model"

        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [mindspore.nn.Dense(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [mindspore.nn.ReLU()]#??
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [mindspore.nn.Dense(mid_num1, mid_num2), mindspore.nn.ReLU()]
                else:
                    modules += [mindspore.nn.Dense(mid_num2, mid_num2), mindspore.nn.ReLU()]
                pre_num = mid_num2
            modules += [mindspore.nn.Dense(pre_num, bit)]
        self.fc = mindspore.nn.SequentialCell(*modules)

        self.norm = norm

    def forward(self, x):
        out1 = self.fc(x)
        out = mindspore.ops.tanh(out1)
        if self.norm:
            norm_x = mindspore.ops.norm(A, ord='fro', dim=1, keepdim=True)
            out = out / norm_x
        return out1,out
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False
