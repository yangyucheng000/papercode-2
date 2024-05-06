import math
import mindspore
from mindspore import nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal

class MLP(nn.Cell):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Dense(dim_in, dim_hidden, weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Dense(dim_hidden, dim_out, weight_init="normal", bias_init="zeros")
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class NIN(nn.Cell):
    def __init__(self, num_classes=10):
        super(NIN, self).__init__()
        self.num_classes = num_classes

        self.features = nn.SequentialCell(
            nn.Conv2d(3, 192, 5, padding=2, pad_mode='pad', has_bias=True, weight_init=Normal(0.05)),
            nn.ReLU(),
            nn.Conv2d(192, 160, 1, has_bias=True, weight_init=Normal(0.05)),
            nn.ReLU(),
            nn.Conv2d(160, 96, 1, has_bias=True, weight_init=Normal(0.05)),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, ceil_mode=True,pad_mode='pad'),
            nn.Dropout(),

            nn.Conv2d(96, 192, 5, padding=2, pad_mode='pad', has_bias=True, weight_init=Normal(0.05)),
            nn.ReLU(),
            nn.Conv2d(192, 192, 1, has_bias=True, weight_init=Normal(0.05)),
            nn.ReLU(),
            nn.Conv2d(192, 192, 1, has_bias=True, weight_init=Normal(0.05)),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2, pad_mode='pad', ceil_mode=True),
            nn.Dropout(),

            nn.Conv2d(192, 192, 3, padding=1, pad_mode='pad', has_bias=True, weight_init=Normal(0.05)),
            nn.ReLU(),
            nn.Conv2d(192, 192, 1, has_bias=True, weight_init=Normal(0.05)),
            nn.ReLU(),
            nn.Conv2d(192, self.num_classes, 1, has_bias=True, weight_init=Normal(0.05)),
            nn.ReLU(),
            nn.AvgPool2d(8, stride=1, pad_mode='pad')
        )

    def construct(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], self.num_classes)
        return x

    
class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, pad_mode='pad'),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def construct(self, x):
        out = ops.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = ops.relu(out)
        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, pad_mode='pad')
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def construct(self, x):
        out = ops.relu(self.bn1(self.conv1(x)))
        out = ops.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = ops.relu(out)
        return out


class ResNet(nn.Cell):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Dense(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(*layers)

    def construct(self, x):
        out = ops.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = ops.avg_pool2d(out, 4)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=100):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

class SmallCNN(nn.Cell):
    def __init__(self):
        super(SmallCNN, self).__init__()

        self.block1_conv1 = nn.Conv2d(3, 64, 3, padding=1, pad_mode='pad')
        self.block1_conv2 = nn.Conv2d(64, 64, 3, padding=1, pad_mode='pad')
        self.block1_pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1_1 = nn.BatchNorm2d(64)
        self.batchnorm1_2 = nn.BatchNorm2d(64)

        self.block2_conv1 = nn.Conv2d(64, 128, 3, padding=1, pad_mode='pad')
        self.block2_conv2 = nn.Conv2d(128, 128, 3, padding=1, pad_mode='pad')
        self.block2_pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm2_1 = nn.BatchNorm2d(128)
        self.batchnorm2_2 = nn.BatchNorm2d(128)

        self.block3_conv1 = nn.Conv2d(128, 196, 3, padding=1, pad_mode='pad')
        self.block3_conv2 = nn.Conv2d(196, 196, 3, padding=1, pad_mode='pad')
        self.block3_pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm3_1 = nn.BatchNorm2d(196)
        self.batchnorm3_2 = nn.BatchNorm2d(196)

        self.activ = nn.ReLU()

        self.fc1 = nn.Dense(196*4*4,256)
        self.fc2 = nn.Dense(256,10)

    def construct(self, x):
        #block1
        x = self.block1_conv1(x)
        x = self.batchnorm1_1(x)
        x = self.activ(x)
        x = self.block1_conv2(x)
        x = self.batchnorm1_2(x)
        x = self.activ(x)
        x = self.block1_pool1(x)

        #block2
        x = self.block2_conv1(x)
        x = self.batchnorm2_1(x)
        x = self.activ(x)
        x = self.block2_conv2(x)
        x = self.batchnorm2_2(x)
        x = self.activ(x)
        x = self.block2_pool1(x)
        #block3
        x = self.block3_conv1(x)
        x = self.batchnorm3_1(x)
        x = self.activ(x)
        x = self.block3_conv2(x)
        x = self.batchnorm3_2(x)
        x = self.activ(x)
        x = self.block3_pool1(x)

        x = x.view(-1,196*4*4)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)

        return x

def small_cnn():
    return SmallCNN()
