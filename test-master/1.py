import os
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore.mindrecord import FileWriter
from matplotlib import pyplot as plt

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
x = np.arange(-5, 5, 0.3)[:32].reshape((32, 1))
y = 5 * x+  0.1 * np.random.normal(loc=0.0, scale=20.0, size=x.shape)
net = nn.Dense(1, 1)
loss_fn = nn.loss.MSELoss()
opt = nn.optim.SGD(net.trainable_params(), learning_rate=0.01)
with_loss = nn.WithLossCell(net, loss_fn)
train_step = nn.TrainOneStepCell(with_loss, opt).set_train()
 
for epoch in range(20):
	loss = train_step(ms.Tensor(x, ms.float32), ms.Tensor(y, ms.float32))
	tt = [x.asnumpy() for x in net.trainable_params()]
	t, u = np.squeeze(tt[0]), np.squeeze(tt[1])
	str1 = np.array_str(t)
	str2 = np.array_str(u)
	print(str1,str2)
	print('epoch: {0}, loss is {1}'.format(epoch, loss))
	with open('E:\\Python\\test.txt', 'a') as f:
		 f.write(str1+' '+str2+'\n ')

wb= [x.asnumpy() for x in net.trainable_params()]
w, b = np.squeeze(tt[0]), np.squeeze(tt[1])

print('The true linear function is y = 5 * x + 0.1')
print('The trained linear model is y = {0} * x + {1}'.format(w, b))


 
plt.scatter(x, y, label='Samples')
plt.plot(x, w * x +  b, c='r', label='True function')
plt.plot(x, 5 * x +  0.1, c='b', label='Trained model')
plt.legend()
plt.show()
 
for i in range(-10, 11, 5):
    print('x = {0}, predicted y = {1}'.format(i, net(ms.Tensor([[i]], ms.float32))))






