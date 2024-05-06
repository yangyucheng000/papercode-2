import mindspore
from mindspore import nn
from mindspore import Tensor, Parameter
import mindspore.ops as ops
from collections import OrderedDict
import numpy as np

EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    for (old_k, old_w), (new_k, new_w) in zip(model.parameters_and_names(), proxy.parameters_and_names()):
        if len(old_w.shape) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = ops.norm(old_w) / (ops.norm(diff_w) + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = set(diff.keys())
    for name, param in model.parameters_and_names():
        if name in names_in_diff:
            data = param + coeff * diff[name]
            param.set_data(data)

def average_diff(cur_diff, new_diff, beta):
    for key in cur_diff.keys():
        cur_diff[key] = beta * cur_diff[key] + (1 - beta) * new_diff[key]
    return cur_diff


class AdvWeightPerturb(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp_simple(self, inputs_adv, cluster_head, steps=1, gamma=None):
        def target_distribution(batch: Tensor) -> Tensor:
            weight = (batch ** 2) / ops.ReduceSum(batch, 0)
            return (weight.t() / ops.ReduceSum(weight, 1)).t()

        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        if gamma is not None:
            self.gamma = gamma
        for _ in range(steps):
            emb = self.proxy.intermediate_forward_simple(inputs_adv, 0)
            x = cluster_head(emb)
            loss = nn.KLDivLoss(reduction='none')(x.log(), target_distribution(x).detach()).sum(-1).mean()
            self.proxy_optim.zero_grad()
            loss.backward()
            self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def calc_awp(self, inputs_adv, steps=1, gamma=None):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        if gamma is not None:
            self.gamma = gamma

        for _ in range(steps):
            x = self.proxy(inputs_adv)
            logsumexp = ops.logsumexp(x, axis=1)
            l_oe = -(x.mean(axis=1) - logsumexp).mean()
            loss = l_oe
            self.proxy_optim.zero_grad()
            loss.backward()
            ops.clip_by_norm(self.proxy.parameter(), 1)
            self.proxy_optim.step()

        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def calc_awp_reg(self, inputs_adv, steps=1, gamma=None):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        if gamma is not None:
            self.gamma = gamma
        for _ in range(steps):
            scale = Parameter(Tensor([1], mindspore.float32), requires_grad=True)
            x = self.proxy(inputs_adv) * scale
            logsumexp = ops.logsumexp(x, axis=1)
            l_oe = -(x.mean(axis=1) - logsumexp).mean() / x.shape[1]
            # 自动梯度计算
            grad_op = nn.GradOperation(get_by_list=True, sens_param=True)
            sens = Tensor([1], mindspore.float32)
            grads = grad_op(self.proxy, self.proxy.trainable_params(), sens)
            r_mr = ops.ReduceSum()(ops.Square()(grads[0]))
            loss = l_oe - r_mr

            # 更新参数
            self.proxy_optim.clear_grad()
            loss.backward()
            self.proxy_optim.step()

    def calc_awp_smooth(self, inputs, targets, loss_fn, steps=1):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        for _ in range(steps):
            outputs = self.proxy(inputs)
            loss = loss_fn(outputs, targets)
            self.proxy_optim.clear_grad()
            loss.backward()
            ops.clip_by_norm(self.proxy.parameter(), 1)
            self.proxy_optim.step()

        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)

    def resotre(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)
