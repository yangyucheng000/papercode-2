import numpy as np
import mindspore as mp
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import composite as C
import mindspore.ops.function.math_func as mf


class DyChLinear(nn.Dense):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True):
        super(DyChLinear, self).__init__(
            in_channels=in_features,
            out_channels=out_features,
            has_bias=bias)
        self.stage = 'supernet'

    def construct(self, x):
        self.running_inc = x.size(1)
        self.running_outc = self.out_features
        weight = self.weight[:,:self.running_inc]
        
        return ops.dense(x, weight, self.bias)
   

class DyChConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, 
        in_ch_static=False, out_ch_static=False,
        stride=1,
        dilation=1,
        group=1,
        bias=True,
        padding=0,
        pad_mode='pad'):
        super(DyChConv2d, self).__init__(in_channels, out_channels, kernel_size, 
            stride=stride, 
            dilation=dilation, 
            group=group, 
            has_bias=bias, 
            padding=padding,
            pad_mode=pad_mode)
        self.channel_ratio = 1.
        self.in_ch_static = in_ch_static
        self.out_ch_static = out_ch_static

        # For calculating FLOPs 
        self.running_inc = self.in_channels if self.in_ch_static else None
        self.running_outc = self.out_channels if self.out_ch_static else None
        self.running_kernel_size = self.kernel_size[0]
        self.running_groups = group

        self.stage = 'supernet'
        self.channel_choice = -1
        self.channel_list = [0., 0.25, 0.5, 0.75, 1.]
    
    def construct(self, x):
        ch_lst_tensor = Tensor.from_numpy(np.array(self.channel_list)).unsqueeze(-1).float() #.to(x.device) # K,1
        if self.stage=='policy':
            assert isinstance(self.channel_choice, Tensor), 'Please set a valid channel_choice first.'
            if not self.in_ch_static:
                num_ch = self.in_channels*ops.matmul(self.channel_choice, ch_lst_tensor)
                num_ch = ops.where(num_ch==0, ops.ones_like(num_ch), num_ch)
                self.running_inc = num_ch.mean().item()
            if not self.out_ch_static:
                num_ch = self.out_channels*ops.matmul(self.channel_choice, ch_lst_tensor)
                num_ch = ops.where(num_ch==0, ops.ones_like(num_ch), num_ch)
                self.running_outc = num_ch.mean().item()
            
            # Training with channel mask
            weight = self.weight
            output = ops.conv2d(x,
                              weight,
                              self.bias,
                              self.stride,
                              self.pad_mode,
                              self.padding,
                              self.dilation,
                              self.group)
            
            if not self.out_ch_static:
                output = apply_differentiable(output, ops.stop_gradient(self.channel_choice), \
                                                self.channel_list, self.out_channels)
        else:
            # For training supernet and deterministic dynamic inference
            if not self.in_ch_static:
                self.running_inc = x.shape[1]
                assert self.running_inc == max(1, int(self.in_channels*self.channel_ratio)), \
                'running input channel %d does not match %dx%.2f'%(self.running_inc, self.in_channels, self.channel_ratio)
            
            # since 0 channel is invalid for calculating output tensors, we use 1 channel to approximate skip operation
            if not self.out_ch_static:
                self.running_outc = max(1, int(self.out_channels*self.channel_ratio))
            weight = self.weight[:self.running_outc,:self.running_inc]
            bias = self.bias[:self.running_outc] if self.bias is not None else None
            output = ops.conv2d(x, weight, bias, self.stride, self.pad_mode, self.padding, self.dilation, self.group)
        
        return output


class DyChBatchNorm2d(nn.Cell):
    def __init__(self, in_planes, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(DyChBatchNorm2d, self).__init__()
        self.in_planes = in_planes
        self.channel_choice = -1
        self.channel_list = [0, 0.25, 0.5, 0.75, 1.]
        self.aux_bn = nn.CellList([
            nn.BatchNorm2d(max(1, int(in_planes*ch)), affine=False) for ch in self.channel_list[:-1]
        ])
        self.aux_bn.append(nn.BatchNorm2d(int(self.channel_list[-1]*in_planes),
                                          eps=eps,
                                          momentum=momentum,
                                          affine=affine))
        self.affine = affine
        self.stage = 'supernet'
    
    def set_zero_weight(self):
        if self.affine:
            nn.init.zeros_(self.aux_bn[-1].weight)
    
    @property
    def weight(self):
        return self.aux_bn[-1].gamma

    @property
    def bias(self):
        return self.aux_bn[-1].beta

    def construct(self, x):
        running_inc = x.shape[1]
        idx_offset = 5-len(self.channel_list)
        if self.stage=='policy':
            assert isinstance(self.channel_choice, Tensor), 'Please set a valid channel_choice first.'
            
            # Training with channel mask
            running_mean = ops.zeros_like(self.aux_bn[-1].running_mean).repeat(len(self.channel_list), 1)
            running_var = ops.zeros_like(self.aux_bn[-1].running_var).repeat(len(self.channel_list), 1)
            for i in range(len(self.channel_list)):
                n_ch = max(1, int(self.in_planes*self.channel_list[i]))
                running_mean[i, :n_ch] += self.aux_bn[i+idx_offset].running_mean
                running_var[i, :n_ch] += self.aux_bn[i+idx_offset].running_var
            running_mean = ops.matmul(self.channel_choice.detach(), running_mean)[..., None, None].expand_as(x)
            running_var = ops.matmul(self.channel_choice.detach(), running_var)[..., None, None].expand_as(x)
            weight = self.weight[:running_inc] if self.affine else None
            bias = self.bias[:running_inc] if self.affine else None
            x = (x - running_mean) / ops.sqrt(running_var + self.aux_bn[-1].eps)
            x = x * weight[..., None, None].expand_as(x) + bias[..., None, None].expand_as(x)
            return apply_differentiable(x, self.channel_choice.detach(), self.channel_list, self.in_planes)
        else:
            running_channel_ratio = 0. if running_inc==1 else running_inc/self.in_planes
            assert running_channel_ratio in self.channel_list, 'Current channel ratio %f is not existed!'%running_channel_ratio
            idx = self.channel_list.index(running_channel_ratio)
            running_mean = self.aux_bn[idx].moving_mean
            running_var = self.aux_bn[idx].moving_variance
            weight = self.aux_bn[-1].gamma[:running_inc] if self.affine else None
            bias = self.aux_bn[-1].beta[:running_inc] if self.affine else None
            return ops.batch_norm(x,
                                running_mean,
                                running_var,
                                weight,
                                bias,
                                self.training,
                                self.aux_bn[-1].momentum,
                                self.aux_bn[-1].eps)


def apply_differentiable(x, channel_choice, channel_list, in_channels, logit=False):
    ret = ops.zeros_like(x)
    for idx in range(len(channel_list)):
        x = ops.where(ops.isnan(x), ops.zeros_like(x), x)
        n_ch = max(1, int(in_channels*channel_list[idx]))
        if logit:
            ret[:, :n_ch] += x[:, :n_ch] * (
                channel_choice[:, idx, None].expand_as(x[:, :n_ch]))
        else:
            ret[:, :n_ch] += x[:, :n_ch] * (
                channel_choice[:, idx, None, None, None].expand_as(x[:, :n_ch]))
    return ret


neg_tensor = P.Neg()
def gumbel_softmax(logits, tau=1, dim=-1):
    mf._check_logits_tensor(logits)
    mf._check_logits_shape(logits)
    logits_dtype = mf._get_cache_prim(P.DType)()(logits)
    mf._check_input_dtype("logits", logits_dtype, [mstype.float16, mstype.float32], "gumbel_softmax")
    mf._check_attr_dtype("tau", tau, [float], "gumbel_softmax")
    mf._check_attr_dtype("dim", dim, [int], "gumbel_softmax")
    mf._check_positive_float(tau, "tau", "gumbel_softmax")
    mf._check_int_range(dim, -1, len(logits.shape), 'dim', "gumbel_softmax")

    log_op = mf._get_cache_prim(P.Log)()
    const_op = mf._get_cache_prim(P.ScalarToTensor)()

    sample_shape = mf._get_cache_prim(P.Shape)()(logits)
    uniform = C.uniform(sample_shape, const_op(
        0.0, mstype.float32), const_op(1.0, mstype.float32))
    uniform = mf._get_cache_prim(P.Cast)()(uniform, logits_dtype)
    gumbel = neg_tensor(log_op(neg_tensor(log_op(uniform))))
    gumbel = (logits + gumbel) / tau
    y_soft = mf._get_cache_prim(P.Softmax)(dim)(gumbel)
    index = y_soft.argmax(axis=dim)
    y_hard = mf._get_cache_prim(P.OneHot)(dim)(index, sample_shape[dim], Tensor(1, logits_dtype),
                                            Tensor(0, logits_dtype))
    ret = ops.stop_gradient(y_hard - y_soft) + y_soft
    
    return y_soft, ret, index
