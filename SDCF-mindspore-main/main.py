import os
from tqdm import tqdm
from sdcom import SDCOM
from dataset import build_anetdataloader
    
import pickle
import numpy as np
import mindspore
from mindspore import context, ops, Tensor
from utils import accuracy, cal_map, AverageMeter
    
    
if __name__=='__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    
    # Load weight
    model = SDCOM()
    # model.set_train(False)
    model.set_stage('inference')
    param_dict = mindspore.load_checkpoint('./checkpoint/sdcom_anet_f16_mindspore.ckpt')
    not_load_list = mindspore.load_param_into_net(model, param_dict)
    
    # Build Dataloader
    val_loader = build_anetdataloader('/mnt/e/datasets/activitynet_pil', bs=2, n_worker=1)
    
    # Evaluation
    all_target = []
    all_results = []
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()
    nbar = tqdm(total=len(val_loader))
    count = 0
    for data in val_loader:
        count+=1
        nbar.update(1)
        input_tensor, target, vid_ids = data[0], data[1], data[2]
        all_target.append(target)
        target = target[:,0]
        out_logit, out_pred, ch_scores, ch_idx, policy_feat, slim_feat = model(input_tensor)
        all_results.append(out_pred)
        
        _b = target.shape[0]
        prec1, prec5 = accuracy(out_pred, target, topk=(1, 5))
        prec1_m.update(prec1, _b)
        prec5_m.update(prec5, _b)
        # if count >= 10:
        #     break 
    nbar.close()
    
    np.save('pred.npz', ops.cat(all_results).asnumpy())
    np.save('gt.npz', ops.cat(all_target).asnumpy())
        
    mAP, _ = cal_map(ops.cat(all_results), ops.cat(all_target)[:, 0:1])
    print('map: %.2f, prec1: %.2f, prec5: %.2f.'%(mAP, prec1_m.avg, prec5_m.avg))

        