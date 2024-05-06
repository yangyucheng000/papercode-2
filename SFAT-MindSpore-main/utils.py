import copy
import mindspore
from mindspore import dataset
from mindspore.dataset import vision,transforms
import mindspore.ops as ops
from sampling import cifar_iid, svhn_iid, cifar_noniid_skew, svhn_noniid_skew, cifar100_noniid_skew, svhn_noniid_unequal, cifar10_noniid_unequal
from PIL import Image
import numpy as np
import os
import pickle
import scipy.io as sio

class CIFAR10(object):
    train_list = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5',
    ]
    test_list = [
        'test_batch',
    ]

    def __init__(self, root, train, transform=None, target_transform=None):
        super(CIFAR10, self).__init__()
        self.root = root
        self.train = train  # training set or test set
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

        for file_name in downloaded_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

class CIFAR100(object):
    def __init__(self, root, train, transform=None, target_transform=None):
        super(CIFAR100, self).__init__()
        self.root = root
        self.train = train  # training set or test set
        if self.train:
            downloaded_list = ['train']
        else:
            downloaded_list = ['test']
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

        for file_name in downloaded_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

class SVHN(object):
    split_list = {
        "train": "train_32x32.mat",
        "test": "test_32x32.mat",
        "extra": "extra_32x32.mat"
    }

    def __init__(self, root, split = "train", transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.filename = self.split_list[split]

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        self.labels = loaded_mat["y"].astype(np.int64).squeeze()

        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):

        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)

def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(image_transforms, 'image')
    if len(dataset.get_col_names())>2:
        dataset = dataset.map(label_transform, 'fine_label')
        dataset = dataset.map(label_transform, 'coarse_label')
    else:
        dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'cifar-10':
        data_dir = './data/cifar-10-batches-py/'

        train_dataset = CIFAR10(data_dir,train=True)
        test_dataset = CIFAR10(data_dir,train=False)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = cifar10_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid_skew(train_dataset, args.num_users)

    elif args.dataset == 'svhn':
        data_dir = '../data/svhn'

        train_dataset = SVHN(data_dir, split='train')
        test_dataset = SVHN(data_dir, split='test')

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = svhn_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = svhn_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                #print(train_dataset.labels)
                user_groups = svhn_noniid_skew(train_dataset, args.num_users)

    elif args.dataset == 'cifar-100':
        data_dir = './data/cifar-100-python'

        train_dataset = CIFAR100(data_dir,train=True)
        test_dataset = CIFAR100(data_dir,train=False)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar100_noniid_skew(train_dataset, args.num_users)


    return train_dataset, test_dataset, user_groups

# FedAvg
def average_weights(w):
    """
    Returns the average of the weights.
    """

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        print(key)
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = mindspore.Parameter(ops.div(w_avg[key], len(w)))
    return w_avg

# # FedAvg unequal
def average_weights_unequal(w, idx_num):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        print(key)
        w_avg[key] = w_avg[key] * float(idx_num[0]*len(w)/sum(idx_num))
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * float(idx_num[i]*len(w)/sum(idx_num))
        w_avg[key] = mindspore.Parameter(ops.div(w_avg[key], len(w)))
    return w_avg
    
# SFAT
def average_weights_alpha(w, lw, idx, p):
    """
    Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        cou = 0
        if (lw[0] >= idx):
            w_avg[key] = w_avg[key] * p
        for i in range(1, len(w)):
            if (lw[i] >= idx) and (('bn' not in key)):
                w_avg[key] = w_avg[key] + w[i][key] * p
            else:
                cou += 1 
                w_avg[key] = w_avg[key] + w[i][key]
        w_avg[key] = mindspore.Parameter(ops.div(w_avg[key], cou+(len(w)-cou)*p))
    return w_avg


# # SFAT unequal
def average_weights_alpha_unequal(w, lw, idx, p, idx_num):
    """
    Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        cou = 0
        if (lw[0] >= idx):
            w_avg[key] = w_avg[key] * p * float(idx_num[0]*len(w)/sum(idx_num))
        else:
            w_avg[key] = w_avg[key] * float(idx_num[0]*len(w)/sum(idx_num))
        for i in range(1, len(w)):
            if (lw[i] >= idx) and (('bn' not in key)):
                w_avg[key] = w_avg[key] + w[i][key] * p * float(idx_num[i]*len(w)/sum(idx_num))
            else:
                cou += 1 
                w_avg[key] = w_avg[key] + w[i][key] * float(idx_num[i]*len(w)/sum(idx_num))
        w_avg[key] = mindspore.Parameter(ops.div(w_avg[key], cou+(len(w)-cou)*p))
    return w_avg  

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
