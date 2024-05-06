import numpy as np
import sys
import argparse
import mindspore
from mindspore import ops, nn
from mindspore import dataset as ds
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
import mindspore.dataset.vision as c_vision
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from models.models2 import WideResNet
import utils.utils_awp as awp
from utils.display_results import get_measures, print_measures
from utils.validation_dataset import validation_split
import mindspore.ops as ops
from mindspore.experimental import optim

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with DOE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default="cifar10", choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=0, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=1, help='Pre-fetching threads.')
# EG specific
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--begin_epoch', type=int, default=0)

args = parser.parse_args()


def set_seed(seed=1):
    mindspore.set_seed(seed)
    np.random.seed(seed)


# mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="CPU", env_config_path="./mindspore_config.json", graph_kernel_flags="--memory_optimize_level=O1")
mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

# set_context(mode=PYNATIVE_MODE)
# 设置CIFAR-10图像的均值和标准差0
if 'cifar' in args.dataset:
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
else:
    mean = Tensor([0.485, 0.456, 0.406], mindspore.float32).reshape(3, 1, 1).asnumpy().tolist()
    std = Tensor([0.229, 0.224, 0.225], mindspore.float32).reshape(3, 1, 1).asnumpy().tolist()

# 定义训练集的数据增强和预处理操作
train_transforms = [
    c_vision.RandomHorizontalFlip(),
    c_vision.RandomCrop(32, padding=4),
    c_vision.Normalize(mean=mean, std=std),
    c_vision.HWC2CHW()
]

# 定义测试集的数据预处理操作
test_transforms = [
    c_vision.Normalize(mean=mean, std=std),
    c_vision.HWC2CHW()
]

# 加载CIFAR数据集
if args.dataset == 'cifar10':
    train_data_in = ds.Cifar10Dataset("../data/cifar-10-batches-bin", usage='train', shuffle=True)
    train_data_in = train_data_in.map(train_transforms, input_columns='image')
    test_data = ds.Cifar10Dataset("../data/cifar-10-batches-bin", usage='test', shuffle=False)
    test_data = test_data.map(test_transforms, input_columns='image')
    cifar_data = ds.Cifar10Dataset("../data/cifar-10-batches-bin", usage='test', shuffle=True)
    cifar_data = cifar_data.map(test_transforms, input_columns='image')
    num_classes = 10
else:
    train_data_in = ds.Cifar100Dataset("../data/cifar-100-batches-bin", usage='train', shuffle=True)
    train_data_in = train_data_in.map(train_transforms, input_columns='image')
    test_data = ds.Cifar100Dataset("../data/cifar-100-batches-bin", usage='test', shuffle=False)
    test_data = test_data.map(test_transforms, input_columns='image')
    cifar_data = ds.Cifar100Dataset("../data/cifar-100-batches-bin", usage='test', shuffle=True)
    cifar_data = cifar_data.map(test_transforms, input_columns='image')
    num_classes = 100

calib_indicator = ''
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'

train_loader_in = train_data_in.batch(args.batch_size, num_parallel_workers=args.prefetch, drop_remainder=True)
test_loader = test_data.batch(args.batch_size, num_parallel_workers=args.prefetch, drop_remainder=True)
cifar_loader = cifar_data.batch(args.test_bs, num_parallel_workers=1, drop_remainder=True)

ood_transforms = [
    c_vision.Decode(),
    c_vision.Resize(32),
    c_vision.RandomHorizontalFlip(),
    c_vision.RandomCrop(32, padding=4),
    c_vision.Normalize(mean=mean, std=std),
    c_vision.HWC2CHW()

]
ood_data = ds.ImageFolderDataset(dataset_dir="../data/tiny-imagenet-200/train", shuffle=True)
ood_data = ood_data.map(ood_transforms, ["image"])
train_loader_out = ood_data.batch(args.oe_batch_size, num_parallel_workers=args.prefetch, drop_remainder=True)

transformls = [
    c_vision.Decode(),
    c_vision.Resize(32),
    c_vision.Normalize(mean=mean, std=std),
    c_vision.HWC2CHW()
]

lsunc_data = ds.ImageFolderDataset(dataset_dir="../data/LSUN", shuffle=True, )
lsunc_data = lsunc_data.map(transformls, input_columns="image")
lsunc_loader = lsunc_data.batch(args.test_bs, num_parallel_workers=1, drop_remainder=True)

lsunr_data = ds.ImageFolderDataset(dataset_dir="../data/LSUN_resize/", shuffle=True, )
lsunr_data = lsunr_data.map(transformls, input_columns="image")
lsunr_loader = lsunr_data.batch(args.test_bs, num_parallel_workers=1, drop_remainder=True)

transformis = [
    c_vision.Decode(),
    c_vision.Normalize(mean=mean, std=std),
    c_vision.HWC2CHW()
]
isun_data = ds.ImageFolderDataset(dataset_dir="../data/iSUN", shuffle=True, )
isun_data = isun_data.map(transformis, input_columns="image")
isun_loader = isun_data.batch(args.test_bs, num_parallel_workers=1, drop_remainder=True)

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.numpy()


class OEAndCELoss(nn.LossBase):

    def construct(self, x, in_set, target):
        l_ce = ops.cross_entropy(x[:len(in_set[0])], target)
        l_oe = - (x[len(in_set[0]):].mean(1) - ops.logsumexp(x[len(in_set[0]):], axis=1)).mean()
        loss = l_ce + l_oe
        return loss


class OELoss(nn.LossBase):

    def construct(self, x, in_set):
        loss = - (x[len(in_set[0]):].mean(1) - ops.logsumexp(x[len(in_set[0]):], axis=1)).mean()
        return loss


class ProxyLoss(nn.LossBase):

    def construct(self, x, in_set):
        loss = (x[len(in_set[0]):].mean(1) - ops.logsumexp(x[len(in_set[0]):], axis=1)).mean()
        return loss


def get_ood_scores(loader, net, in_dist=False):
    _score = []
    net.set_train(False)
    batch_idx = 0
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= ood_num_examples // args.test_bs and not in_dist:
            break
        output = net(data)
        # smax = ops.Softmax(axis=1)(output)
        smax = output.asnumpy()
        _score.append(-np.max(smax, axis=1))

    if in_dist:
        return concat(_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def get_and_print_results(ood_loader, net, in_score, num_to_avg=args.num_to_avg):
    net.set_train(False)
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader, net)
        if args.out_as_pos:
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0])
        auprs.append(measures[1])
        fprs.append(measures[2])
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)
    print_measures(auroc, aupr, fpr, '')
    return fpr, auroc, aupr


def train(epoch, diff, net, train_loader_in, train_loader_out, optimizer):
    net.set_train()

    def oe_ce_forward_fn(x, in_set, target):
        output = net(x)
        loss = OEAndCELoss()(output, in_set, target)
        return loss

    def oe_forward_fn(x, in_set):
        output = net(x)
        loss = OELoss()(output, in_set)
        return loss

    oe_ce_grad_fn = mindspore.value_and_grad(oe_ce_forward_fn, None, optimizer.parameters)
    oe_grad_fn = mindspore.value_and_grad(oe_forward_fn, None, optimizer.parameters)

    proxy = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=0)
    proxy.set_train()
    proxy_optim = optim.SGD(proxy.trainable_params(), lr=1)
    proxy_loss = ProxyLoss()

    def proxy_scale_forward_fn(x, scale, in_set):
        x = x * scale
        return proxy_loss(x, in_set)

    def proxy_forward_fn(data, in_set):
        x = proxy(data)
        scale = Parameter(Tensor([1], mindspore.float32), requires_grad=True)
        scale_grad_fn = mindspore.grad(proxy_scale_forward_fn, (1))
        scale_grad = scale_grad_fn(x, scale, in_set)
        reg_sur = (scale_grad[0] ** 2).sum()
        return reg_sur

    proxy_grad_fn = mindspore.value_and_grad(proxy_forward_fn, None, proxy_optim.parameters)

    def forward_step2(data, in_set, target):
        x = net(data)
        l_ce = ops.cross_entropy(x[:len(in_set[0])], target)
        return l_ce

    step2_grad_fn = mindspore.value_and_grad(forward_step2, None, optimizer.parameters)

    loss_avg = 0.0
    train_loader_out.offset = np.random.randint(len(train_loader_in))
    for batch_idx, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):

        # 测试，每个epoch只训练20个batch
        # if batch_idx > 20:
        #     break

        data = ops.cat((in_set[0], out_set[0]), axis=0)
        target = in_set[1].int()

        if epoch >= args.warmup:
            gamma = Tensor(np.random.choice([1e-1, 1e-2, 1e-3, 1e-4]), mstype.float32)
            net_params = {param.name: param for param in net.get_parameters()}
            mindspore.load_param_into_net(proxy, net_params)

            loss, grads = proxy_grad_fn(data, in_set)
            # grads = ops.clip_by_norm(grads, 1)
            proxy_optim(grads)

            if epoch == args.warmup and batch_idx == 0:
                diff = awp.diff_in_weights(net, proxy)
            else:
                # diff = awp.diff_in_weights(net, proxy)
                diff = awp.average_diff(diff, awp.diff_in_weights(net, proxy), beta=.6)

            awp.add_into_weights(net, diff, coeff=gamma)

        if epoch < args.warmup:
            loss, grads = oe_ce_grad_fn(data, in_set, target)
        else:
            loss, grads = oe_grad_fn(data, in_set)

        # grads = ops.clip_by_norm(grads, 1)
        optimizer(grads)

        if epoch >= args.warmup:
            awp.add_into_weights(net, diff, coeff=-gamma)
            loss, grads = step2_grad_fn(data, in_set, target)
            optimizer(grads)

        loss_avg = loss_avg * 0.8 + float(loss.item()) * 0.2
        # loss_avg = loss_avg * 0.8 + loss.asnumpy().mean() * 0.2
        scheduler.step()
        sys.stdout.write(f'\r epoch {epoch} {batch_idx + 1}/{len(train_loader_in)} loss {loss_avg:.2f}')

    print()
    return diff


def test(net, data_loader):
    net.set_train(False)
    correct = 0
    sum = 0
    for data, target in data_loader:
        output = net(data)
        pred = np.argmax(output.asnumpy(), axis=1)
        correct += (pred == target.asnumpy()).sum()
        sum += len(target)
    return correct / sum * 100


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


print('Beginning Training\n')
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
# Restore model
if args.dataset == 'cifar10':
    model_path = 'ckpt/cifar10_wrn_pretrained_epoch_99.pt'
else:
    model_path = 'ckpt\\cifar100_wrn_pretrained_epoch_99.ckpt'
import pandas as pd
import torch

def load_from_torch(ckpt_path, model):
    prams_ms = model.parameters_dict().keys()
    prams_ms_lst = pd.DataFrame(prams_ms)

    pt_values_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    ms_values_dict = {}

    for i in prams_ms_lst.values:
        ms_key = i.item()
        pt_key = i.item()
        if "bn" in ms_key:
            if "moving_mean" in ms_key:
                pt_key = pt_key.replace("moving_mean", "running_mean")
            elif "moving_variance" in ms_key:
                pt_key = pt_key.replace("moving_variance", "running_var")
            elif "gamma" in ms_key:
                pt_key = pt_key.replace("gamma", "weight")
            elif "beta" in ms_key:
                pt_key = pt_key.replace("beta", "bias")

        if "layers" in ms_key:
            pt_key = pt_key.replace("layers", "layer")

        pt_val = pt_values_dict[pt_key].data.numpy()
        ms_val = Parameter(pt_val, ms_key)
        # print(ms_val)
        ms_values_dict[ms_key] = ms_val
    mindspore.load_param_into_net(net, ms_values_dict)
    return net

net = load_from_torch(model_path, net)


optimizer = optim.SGD(net.trainable_params(), lr=args.learning_rate, momentum=args.momentum,
                      weight_decay=args.decay,
                      nesterov=True)
lr_lambda = lambda step: cosine_annealing(step, args.epochs * len(train_loader_in), 1, 1e-6 / args.learning_rate)
scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)

diff = None

print('  FPR95 AUROC AUPR (acc %.2f)' % test(net, test_loader))
for epoch in range(args.begin_epoch, args.epochs):
    diff = train(epoch, diff, net, train_loader_in, train_loader_out, optimizer)

print('  FPR95 AUROC AUPR (acc %.2f)' % test(net, test_loader))

in_score = get_ood_scores(test_loader, net, in_dist=True, )
metric_ll = []
print('lsun')
metric_ll.append(get_and_print_results(lsunc_loader, net, in_score))
print('isun')
metric_ll.append(get_and_print_results(isun_loader, net, in_score))
# print('texture')
# metric_ll.append(get_and_print_results(texture_loader, in_score))
print('total')
print('& %.2f & %.2f & %.2f' % tuple((100 * Tensor(metric_ll).mean(0)).tolist()))
print('cifar')
get_and_print_results(cifar_loader, net, in_score)
