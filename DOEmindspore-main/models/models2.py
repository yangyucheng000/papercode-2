import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal, Zero
from mindspore import Tensor
import numpy as np


class BasicBlock(nn.Cell):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, pad_mode='pad', has_bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, pad_mode='pad', has_bias=False)
        self.drop_rate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                       padding=0, pad_mode='pad', has_bias=False)
                             if not self.equalInOut else None)

    def construct(self, x):
        # print(x.shape)
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(x if not self.equalInOut else out)
        if self.drop_rate > 0:
            # out = nn.Dropout(1 - self.drop_rate)(out)
            out = nn.Dropout(p=self.drop_rate)(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if not self.equalInOut:
            return out + self.convShortcut(x)
        else:
            return out + x


class NetworkBlock(nn.Cell):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layers = nn.SequentialCell([
            block(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, dropRate)
            for i in range(nb_layers)
        ])

    def construct(self, x):
        # print(x.shape)
        return self.layers(x)


class WideResNet(nn.Cell):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0), 'Depth must be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, pad_mode='pad', has_bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU()
        self.fc = nn.Dense(nChannels[3], num_classes)
        # self.fc = nn.Dense(128*8*8, num_classes)

        self.nChannels = nChannels[3]

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                cell.weight.set_data(Normal(1.0 / n)(cell.weight.shape))  # 初始化权重
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(Normal(1, 0.01)(cell.gamma.shape))  # 初始化gamma
                cell.beta.set_data(Zero()(cell.beta.shape))  # 初始化beta
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(Normal(1.0 / np.sqrt(cell.in_channels))(cell.weight.shape))  # 初始化权重

    def construct(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)
        out = self.relu(out)
        # out = nn.AvgPool2d(8)(out)
        out = nn.MaxPool2d(8)(out)
        out = out.view(out.shape[0], -1)
        return self.fc(out)

# input_tensor = Tensor(np.random.rand(1, 3, 32, 32), mindspore.float32)
# model = WideResNet(depth=28,num_classes=10,widen_factor=2,dropRate=0.0)
# print((model(input_tensor)).shape)
