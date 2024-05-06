import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import numpy as np
from numpy.random import randint

from mindspore import ops
from mindvideo.data import transforms
import mindspore.dataset.transforms as transf
from mindspore.dataset import GeneratorDataset
# from mindspore.dataset.transforms.py_transforms import Compose


class AnetDataset():
    def __init__(self, data_root='', split='val'):
        self.name = 'anet'
        self.n_segment=16
        self.new_length = 1
        self.num_classes = 200
        
        anno_file = './pre_processing/anet/anet_%s_split.txt'
        class_file = './pre_processing/anet/classInd.txt'
        with open(class_file) as f:
            lines = f.readlines()
        self.id_2_class = {line.strip().split(',')[0]: line.strip().split(',')[1] for line in lines}
        self.class_2_id = {v: k for k, v in self.id_2_class.items()}
        
        print('Loading annotations...')
        self.anno = []
        with open(anno_file % split) as f:
            lines = f.readlines()
            miss = 0
            nbar = tqdm(total=len(lines))
            for n_l, line in enumerate(lines):
                nbar.update(1)
                n_l+=1
                items = line.strip().split(',')
                vid_id = items[0]
                if int(items[1])<=3:
                    continue
                vid_path = os.path.join(data_root, vid_id[2:]+'.pkl')
                if not os.path.isfile(vid_path):
                    miss+=1
                    continue
                # labels = ops.Tensor([-1, -1, -1])
                labels = np.array([-1,-1,-1])
                objs = sorted(list(set([int(x) for x in items[2:]])))
                for i,l in enumerate(objs):
                    labels[i] = l

                if labels[-2] > -1:
                    if labels[-1] > -1:
                        np.random.shuffle(labels)
                    else:
                        if np.random.rand(1) > 0.5:
                            labels = labels[[0,1,2]]
                        else:
                            labels = labels[[1,0,2]]
                assert labels[0] > -1
                
                self.anno.append({'id': vid_id, 'path': vid_path, 'label': labels})
                # if n_l==500:
                #     break
                # self.anno.append({'id': vid_id, 'path': vid_path, 'label': label, 'nf': len(os.listdir(vid_path))})
            nbar.close()
        #self.anno = self.anno[:1000]
        print('Creating %s dataset completed, %d samples, miss: %d.' % (split, len(self.anno), miss))
    
    def _get_val_indices(self, num_frames, need_f):
        if num_frames > need_f + self.new_length - 1:
            tick = (num_frames-self.new_length+1) / float(need_f)
            # offsets = np.array([int(tick * x) for x in range(need_f)])
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(need_f)])
        else:
            offsets = np.zeros((need_f,))
            #offsets = np.array(list(range(num_frames-self.new_length+1)) + [num_frames-self.new_length] * (need_f-num_frames+self.new_length-1))
        return offsets + 1
    
    def get_indices(self, nf, need_f):
        indices = self._get_val_indices(nf, need_f)
        return indices
    
    def __len__(self):
        return len(self.anno)
    
    def __getitem__(self, index):
        vid_info = self.anno[index]
        label = vid_info['label']
        video_pkl = pickle.load(open(vid_info['path'],'rb'))
        nf = len(video_pkl)
        indices = [x-1 for x in self.get_indices(nf, self.n_segment)]
        if len(indices)!=self.n_segment:
            raise KeyError(vid_info['path']+' frame number:%d' % nf)
        indices = self.ext_indices(indices, nf)
        img_group = [np.array(Image.open(video_pkl[ind]).convert('RGB')) for ind in indices]
        img_group = np.stack(img_group, axis=0)
        return img_group, label, vid_info['id']
    
    def ext_indices(self, indices, num_frames):
        new_indices = []
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                new_indices.append(p)
                if p+1 < num_frames:
                        p += 1
        return new_indices

def build_anetdataloader(data_root='/mnt/e/datasets/activitynet_pil', bs=1, n_worker=4, repeat_num=1):
    dataset = AnetDataset(data_root)
    trans = [
            transforms.VideoShortEdgeResize(256),
            transforms.VideoCenterCrop((224,224)),
            transforms.VideoRescale(shift=0),
            transforms.VideoReOrder(order=(3, 0, 1, 2)),
            transforms.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    dataloader = GeneratorDataset(
        dataset, 
        column_names=['video','label','id'], 
        shuffle=False,
        num_parallel_workers=n_worker,
    )
    dataloader = dataloader.map(operations=trans,
                                input_columns='video',
                                num_parallel_workers=n_worker)
    dataloader = dataloader.batch(bs, drop_remainder=True)
    dataloader = dataloader.repeat(repeat_num)
    return dataloader
    

if __name__=='__main__':
    dset = build_anetdataloader()
    for data in dset:
        print(data[-1])
    