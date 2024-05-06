import os
import time
import math
import numpy as np
from copy import deepcopy
from collections import namedtuple
from collections import OrderedDict

import torch
import mindspore
from mindspore import Tensor, ops
from mindspore.train.serialization import save_checkpoint


def set_exist_attr(m, attr, value):
    if hasattr(m, attr):
        setattr(m, attr, value)


def load_org_weights(model, pretrained, repeat_bn=True):
    pre_dict = torch.load(pretrained, map_location='cpu')
    tic = time.time()
    model_dict = model.state_dict()
    if 'state_dict' in pre_dict.keys():
        pre_dict = pre_dict['state_dict']
    for name in model_dict.keys():
        if 'num_batches_tracked' in name:
            continue
        is_null = True
        # TODO: Load BatchNorm Layers
        if repeat_bn and 'aux_bn' in name:
            tmp_name = name.split('aux_bn')[0]+name.split('.')[-1]
        else:
            tmp_name = name

        try:
            if model_dict[name].shape == pre_dict[tmp_name].shape:
                model_dict[name] = pre_dict[tmp_name]
                is_null = False
            else:
                print('size mismatch for %s, expect (%s), but got (%s).'
                        % (name, ','.join([str(x) for x in model_dict[name].shape]),
                            ','.join([str(x) for x in pre_dict[tmp_name].shape])))
            continue
        except KeyError:
            pass
        if is_null:
            print('Do not load %s' % name)

    model.load_state_dict(model_dict)
    print('Load pre-trained weightin %.4f sec.' % (time.time()-tic))


def pytorch2mindspore(torch_model_path, save_mm_model_pth):
    bn_list = [
        'policy_net.feat_net.features.0.1',
        'policy_net.feat_net.features.1.conv.0.1',
        'policy_net.feat_net.features.1.conv.2',
        'policy_net.feat_net.features.2.conv.0.1',
        'policy_net.feat_net.features.2.conv.1.1',
        'policy_net.feat_net.features.2.conv.3',
        'policy_net.feat_net.features.3.conv.0.1',
        'policy_net.feat_net.features.3.conv.1.1',
        'policy_net.feat_net.features.3.conv.3',
        'policy_net.feat_net.features.4.conv.0.1',
        'policy_net.feat_net.features.4.conv.1.1',
        'policy_net.feat_net.features.4.conv.3',
        'policy_net.feat_net.features.5.conv.0.1',
        'policy_net.feat_net.features.5.conv.1.1',
        'policy_net.feat_net.features.5.conv.3',
        'policy_net.feat_net.features.6.conv.0.1',
        'policy_net.feat_net.features.6.conv.1.1',
        'policy_net.feat_net.features.6.conv.3',
        'policy_net.feat_net.features.7.conv.0.1',
        'policy_net.feat_net.features.7.conv.1.1',
        'policy_net.feat_net.features.7.conv.3',
        'policy_net.feat_net.features.8.conv.0.1',
        'policy_net.feat_net.features.18.1',
        ] 
            
    par_dict = torch.load(torch_model_path, map_location='cpu')['state_dict']
    new_params_list = []
    for name in par_dict:
        if 'num_batches_tracked' in name:
                continue
        param_dict = {}
        parameter = Tensor.from_numpy(par_dict[name].numpy())
        name = name.replace('running_mean','moving_mean')
        name = name.replace('running_var','moving_variance')
        if 'bn' in name or 'final_classifier.mlp.layers.1' in name:
            name = name.replace('weight', 'gamma')
            name = name.replace('bias', 'beta')
        if 'policy_net' in name:
            if 'conv.0.1.' in name or 'conv.1.1.' in name or 'conv.3.' in name \
                or 'features.1.conv.2.' in name or 'features.0.1.' in name \
                or 'features.18.1.' in name or 'gate.encoder.1.' in name \
                or 'gate.encoder.5.' in name:
                name = name.replace('weight', 'gamma')
                name = name.replace('bias', 'beta')
        
        param_dict['name'] = name
        param_dict['data'] = parameter
        new_params_list.append(param_dict)
    save_checkpoint(new_params_list, save_mm_model_pth)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    # pred = pred.t()
    correct = pred.equal(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.numpy()*(100.0 / batch_size))
    return res


def get_multi_hot(test_y, classes, assumes_starts_zero=True):
    bs = test_y.shape[0]
    label_cnt = 0

    # TODO ranking labels: (-1,-1,4,5,3,7)->(4,4,1,2,0,3)
    if not assumes_starts_zero:
        for label_val in torch.unique(test_y):
            if label_val >= 0:
                test_y[test_y == label_val] = label_cnt
                label_cnt += 1

    gt = ops.zeros((bs, classes + 1))  # TODO(yue) +1 for -1 in multi-label case
    for i in range(bs):
        for j in range(test_y.shape[1]):
            gt[i, test_y[i, j]] = 1  # TODO(yue) see?

    return gt[:, :classes]


def cal_map(output, old_test_y, is_soft=False):
    batch_size = output.shape[0]
    num_classes = output.shape[1]
    ap = np.zeros(num_classes)
    test_y = old_test_y

    if is_soft:
        gt = test_y
    else:
        gt = get_multi_hot(test_y, num_classes)

    probs = ops.softmax(output, axis=1)

    rg = ops.arange(1, batch_size+1).float()
    for k in range(num_classes):
        scores = probs[:, k]
        targets = gt[:, k]
        _, sortind = ops.sort(scores, 0, True)
        truth = targets[sortind]
        tp = truth.float().cumsum(0)
        precision = tp.div(rg)
        # ap[k] = precision[truth.byte()].sum() / max(float(truth.sum()), 1)
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap.mean() * 100, ap * 100


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


if __name__=='__main__':
    pytorch2mindspore('./checkpoint/tslimnet_anet_f16_torch.pth', './checkpoint/sdcom_anet_f16_mindspore.ckpt')
    from sdcom import SDCOM
    model = SDCOM()
    param_dict = mindspore.load_checkpoint('./checkpoint/sdcom_anet_f16_mindspore.ckpt')
    not_load_list = mindspore.load_param_into_net(model, param_dict)
    print('OK.')
    