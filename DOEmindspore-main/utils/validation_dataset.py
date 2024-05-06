import mindspore.dataset as ds
import numpy as np


class PartialDataset(ds.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.parent_ds[item + self.offset]


def validation_split(dataset, val_share=0.1):
    val_offset = int(len(dataset) * (1 - val_share))
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset) - val_offset)


class PartialFolder(ds.Dataset):
    def __init__(self, parent_ds, perm, length):
        self.parent_ds = parent_ds
        self.perm = perm
        self.length = length
        super(PartialFolder, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.parent_ds[self.perm[item]]


def validation_split_folder(dataset, val_share=0.1):
    num_train = int(len(dataset) * (1 - val_share))
    num_val = len(dataset) - num_train
    perm = np.asarray(range(len(dataset)))
    np.random.seed(0)
    np.random.shuffle(perm)
    train_perm, val_perm = perm[:num_train], perm[num_train:]

    return PartialFolder(dataset, train_perm, num_train), PartialFolder(dataset, val_perm, num_val)
