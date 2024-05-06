# import torch
# from torch import nn
import mindspore
# from torch.nn import functional as F
class ImgModule(mindspore.nn.Cell):
    def __init__(self, pretrain_model=None):
        super(ImgModule, self).__init__()
        # self.features = nn.Sequential(
        self.features = mindspore.nn.SequentialCell(
            # 0 conv1
            mindspore.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            # 1 relu1
            mindspore.nn.ReLU(),
            # 2 norm1
            # nn.LocalResponseNorm(size=2, k=2),
            mindspore.nn.LRN(depth_radius=1,bias=2),
            # 3 pool1

            mindspore.nn.ZeroPad2d((0, 1, 0, 1)),
            mindspore.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 4 conv2
            mindspore.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 5 relu2
            mindspore.nn.ReLU(),
            # 6 norm2
            mindspore.nn.LRN(depth_radius=1,bias=2),
            # 7 pool2
            mindspore.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 8 conv3
            mindspore.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 9 relu3
            mindspore.nn.ReLU(),
            # 10 conv4
            mindspore.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 11 relu4
            mindspore.nn.ReLU(),
            # 12 conv5
            mindspore.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 13 relu5
            mindspore.nn.ReLU(),
            # 14 pool5
            mindspore.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # 15 full_conv6
            mindspore.nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            # 16 relu6
            mindspore.nn.ReLU(),
            # 17 full_conv7
            mindspore.nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            # 18 relu7
            mindspore.nn.ReLU(),
        )
        # fc8
        self.mean = mindspore.ops.zeros((3, 224, 224))
        self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        self.mean = mindspore.Tensor(data['normalization'][0][0][0].transpose()).type(mindspore.float32)
        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, mindspore.nn.Conv2d):#??
                if k > 1:
                    k -= 1
                v.weight.data = mindspore.Tensor(weights[k][0][0][0][0][0].transpose())
                v.bias.data = mindspore.Tensor(weights[k][0][0][0][0][1].reshape(-1))

    def forward(self, x):
        x = x - self.mean.cuda()
        x = self.features(x)
        x = x.squeeze()
        return x

class ImageNet(mindspore.nn.Cell):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=3):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(ImageNet, self).__init__()
        self.module_name = "img_model"

        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [mindspore.nn.Dense(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [mindspore.nn.ReLU()]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [mindspore.nn.Dense(mid_num1, mid_num2), mindspore.nn.ReLU()]
                else:
                    modules += [mindspore.nn.Dense(mid_num2, mid_num2), mindspore.nn.ReLU()]
                pre_num = mid_num2
            modules += [mindspore.nn.Dense(pre_num, bit)]
        self.fc = mindspore.nn.SequentialCell(*modules)
        #self.apply(weights_init)
        self.norm = norm

    def forward(self, x):
        out1 = self.fc(x)
        # out = torch.tanh(out1)
        out = mindspore.ops.tanh(out1)
        if self.norm:
            norm_x = mindspore.ops.norm(out, ord='fro', dim=1, keepdim=True)
            out = out / norm_x
        return out1,out
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False


# class ImageNet(nn.Module):
#     def __init__(self, code_len):
#         super(ImageNet, self).__init__()
#         self.fc1 = nn.Linear(4096, 4096)
#         self.fc_encode = nn.Linear(4096, code_len)

#         self.alpha = 1.0
#         self.dropout = nn.Dropout(p=0.5)
#         self.relu = nn.ReLU(inplace=True)
#        # torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std= 0.1)  

#     def forward(self, x):

#         x = x.view(x.size(0), -1)

#         feat1 = self.relu(self.fc1(x))
#         #feat1 = feat1 + self.relu(self.fc2(self.dropout(feat1)))
#         hid = self.fc_encode(self.dropout(feat1))
#         code = torch.tanh(hid)

#         return code
