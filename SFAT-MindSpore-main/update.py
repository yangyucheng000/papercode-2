import copy
import mindspore
import numpy as np
import attack_generator as attack
from mindspore import nn
from mindspore.dataset import GeneratorDataset
import mindspore.ops as ops
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision

class DatasetSplit(object):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image,label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, alg, anchor, anchor_mu, local_rank, aysn=False, method='AT'):
        self.args = args
        self.trainloader, self.testloader = self.train_test(
            dataset, list(idxs))
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.alg = alg
        self.anchor = anchor
        self.anchor_mu = anchor_mu
        self.local_rank = local_rank
        self.asyn = aysn
        self.method = method

    def train_test(self, dataset, idxs):
        """
        Returns train and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(1.0*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        composed = transforms.Compose(
            [
                vision.Rescale(1.0 / 255.0, 0),
                vision.HWC2CHW()
            ]
        )

        trainloader = GeneratorDataset(source=DatasetSplit(dataset, idxs_train),column_names=['data','labels'], shuffle=True).map(composed,'data').batch(self.args.local_bs)
        testloader = GeneratorDataset(source=DatasetSplit(dataset, idxs_test),column_names=['data','labels'], shuffle=False).map(composed,'data').batch(int(len(idxs_test)/10))
        return trainloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.set_train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = nn.SGD(model.trainable_params(), learning_rate=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = nn.Adam(model.trainable_params(), learning_rate=self.args.lr,
                                         weight_decay=1e-4)

        def forward_fn(data, label):
            logits = model(data)
            loss = self.criterion(logits, label)
            return loss, logits

        grad_fn = mindspore.value_and_grad(forward_fn, None, weights=optimizer.parameters, has_aux=True)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                (loss, log_probs), grads = grad_fn(images, labels)
                optimizer(grads)

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader) * self.args.local_bs,
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.parameters_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_at(self, model, global_round):
        
        epoch_loss = []
        index = 0.0
        index_pgd =0
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = nn.SGD(model.trainable_params(), learning_rate=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = nn.Adam(model.trainable_params(), learning_rate=self.args.lr,
                                         weight_decay=1e-4)

        def forward_fn(data, label):
            logits = model(data)
            loss = self.criterion(logits, label)
            return loss, logits

        grad_fn = mindspore.value_and_grad(forward_fn, None, weights=optimizer.parameters, has_aux=True)

        timestep = 0
        max_local_train = self.args.local_ep
        tau = 0
        
        if self.args.dataset == 'cifar-10':
            eps = 8/255
            sts = 2/255
        if self.args.dataset == 'svhn':
            eps = 4/255
            sts = 1/255
        if self.args.dataset == 'cifar-100':
            eps = 8/255
            sts = 2/255   

        total, correct = 0.0, 0.0
        lop = self.args.local_ep
        
        for iter in range(lop):
            batch_loss = []
            index = 0.0
            index_pgd = 0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if self.method == 'AT':
                    x_adv, ka = attack.PGD(model,images,labels,eps,sts,self.args.num_steps,loss_fn="cent",category="Madry",rand_init=True)
                elif self.method == 'ST':
                    x_adv, ka = images, 0
                  
                model.set_train()
                (loss, log_probs), grads = grad_fn(x_adv, labels)
                
                _, pred_labels = ops.max(log_probs, 1)
                pred_labels = pred_labels.view(-1)
                correct += ops.sum(ops.equal(pred_labels, labels)).item()
                total += len(labels)

                if self.method == 'ST':
                    ka -= loss.sum().item()
                    
                index_pgd += sum(ka)
                ka = -(loss.sum().item())*len(x_adv)
                
                if self.alg == 'FedProx' and global_round > 0:
                    proximal_term = 0
                    for key, key_t in zip(model.parameters_dict(), self.anchor.parameters_dict()) :
                        # update the proximal term 
                        #proximal_term += torch.sum(torch.abs((w-w_t)**2))
                        proximal_term += (model.parameters_dict()[key]-self.anchor.parameters_dict()[key_t]).norm(2)
                    loss = loss + (self.anchor_mu/2)*proximal_term

                optimizer(grads)

                if self.method == 'ST':
                    index += ka
                else:
                    index = index + ka
  
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader) * self.args.local_bs,
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
                
                    
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if timestep == 0 and len(self.local_rank) >= 1 and self.asyn == True and global_round>0:
                selfrank = index / len(self.trainloader)
                ipx_sorted = np.sort(self.local_rank)
                ipx_idx = int(self.args.num_users * self.args.frac)
                ipx_value = ipx_sorted[int(ipx_idx*(1-min(1,global_round/(0.15*self.args.epochs))))]
                if selfrank < ipx_value and max_local_train == self.args.local_ep:
                    print(1)
                    max_local_train = int(self.args.es*max_local_train)
            timestep = timestep+1
            if timestep > max_local_train and len(self.local_rank) >= 1 and self.asyn == True and max_local_train != self.args.local_ep:
                break

        return model.parameters_dict(), sum(epoch_loss) / len(epoch_loss), index / len(self.trainloader), correct/total, index_pgd/ len(self.trainloader)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.set_train(False)
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = ops.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += ops.sum(ops.equal(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss
