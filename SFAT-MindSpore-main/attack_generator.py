import mindspore
import mindspore.ops as ops
import numpy as np
from models import *
from mindspore.dataset import GeneratorDataset

def PGD(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.set_train(False)
    Kappa = ops.ones(len(data))
    if category == "trades":
        x_adv = data + 0.001 * ops.randn(data.shape) if rand_init else data
        nat_output = model(data)
    elif category == "Madry":
        x_adv = data + mindspore.Tensor(np.random.uniform(-epsilon, epsilon, data.shape)).float() if rand_init else data
        x_adv = ops.clamp(x_adv, 0.0, 1.0)
    if loss_fn == "cent":
        loss_adv = nn.CrossEntropyLoss(reduction="mean")
    elif loss_fn == "kl":
        loss_adv = nn.KLDivLoss()
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_adv(logits, label)
        return loss, logits
    grad_fn = mindspore.value_and_grad(forward_fn, grad_position=0, weights=None, has_aux=True)
    for k in range(num_steps):
        (loss, output), grads = grad_fn(x_adv, target)

        predict = ops.max(output, axis=1)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1

        eta = step_size * grads.sign()
        # Update adversarial data
        x_adv = x_adv + eta
        x_adv = ops.minimum(ops.maximum(x_adv, data - epsilon), data + epsilon)
        x_adv = ops.clamp(x_adv, 0.0, 1.0)
    #x_adv = mindspore.Tensor(x_adv)
    return x_adv, Kappa

def eval_clean(model, test_loader, bs):
    model.set_train(False)
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += ops.cross_entropy(output, target).item()
        pred = ops.max(output, 1, keepdims=True)[1]
        correct += pred.equal(target.view_as(pred)).sum().item()
    test_loss /= (len(test_loader) * bs)
    test_accuracy = correct / (len(test_loader)* bs)
    return test_loss, test_accuracy

def eval_robust(model, test_loader, bs, perturb_steps, epsilon, step_size, loss_fn, category, random):
    model.set_train(False)
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        x_adv, _ = PGD(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=random)
        output = model(x_adv)
        test_loss += ops.cross_entropy(output, target).item()
        pred = ops.max(output, 1, keepdims=True)[1]
        correct += pred.equal(target.view_as(pred)).sum().item()
    test_loss /= (len(test_loader) * bs)
    test_accuracy = correct / (len(test_loader) * bs)
    return test_loss, test_accuracy

