import os
import copy
import time
import mindspore
import pickle
import numpy as np
import attack_generator as attack
from models import *
from tqdm import tqdm
from logger import Logger
from options import args_parser
from matplotlib.pyplot import title
from update import LocalUpdate
from utils import get_dataset, average_weights, exp_details, average_weights_alpha
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision

# Save checkpoint
def save_checkpoint(model, checkpoint='./SFAT_result', filename='checkpoint.ckpt'):
    filepath = os.path.join(checkpoint, filename)
    mindspore.save_checkpoint(model, filepath)

if __name__ == '__main__':
    
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()
    exp_details(args)
    
    seed = args.seed
    mindspore.set_seed(seed)
    np.random.seed(seed)

    # Store path
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    global_best_natural = 0
    global_best_pgd = 0
    best_epoch = 0

    print('==> SFAT')
    title = 'SFAT'
    logger_test = Logger(os.path.join(args.out_dir, 'log_results.txt'), title=title)
    logger_test.set_names(['Global Epoch', 'Local Epoch', 'Epoch', 'Natural Test Acc', 'PGD20 Acc'])

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    composed = transforms.Compose(
        [
            vision.Rescale(1.0 / 255.0, 0),
            vision.HWC2CHW()
        ]
    )
    test_loader = GeneratorDataset(source=test_dataset, column_names=['data', 'labels'],
                                   shuffle=False).map(composed,'data').batch(args.local_bs)

    # build model
    if args.modeltype == 'NIN':
        global_model = NIN()
    elif args.modeltype == 'SmallCNN':
        global_model = SmallCNN()
    elif args.modeltype == 'resnet18':
        global_model = ResNet18()

    # set config for pgd
    if args.dataset == 'cifar-10':
        eps = 8/255
        sts = 2/255
    if args.dataset == 'svhn':
        eps = 4/255
        sts = 1/255
    if args.dataset == 'cifar-100':
        eps = 8/255
        sts = 2/255

    # Set the model to train and send it to device.
    print(global_model)

    client_model = [copy.deepcopy(global_model) for i in range(args.num_users)]
    
    # copy weights
    global_weights = global_model.parameters_dict()

    # Training
    train_loss, train_accuracy = [], []
    print_every = 2
    ipx = []

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, idt = [], [], []
        idx_train_acc = []
        ipp = []
        idx_num = []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        ctr = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], alg=args.agg_opt, anchor=global_model, anchor_mu=args.mu, local_rank=ipx, method=args.train_method)
            client_model[idx] = copy.deepcopy(global_model)

            w, loss, ide, idx_train, pp_index = local_model.update_weights_at(
            model=copy.deepcopy(client_model[idx]), global_round=epoch)
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            idt.append(ide)
            #idx_num.append(len(user_groups[idx]))
            #idt.append(ide*(idx_num[ctr]))
            #ctr = ctr+1
            ipp.append(pp_index)
            idx_train_acc.append(idx_train)
        
        ipx = idt
        # update global weights
        # global_weights = average_weights(local_weights) #FedAvg
        global_model.set_train()

        # aggregation methods
        if args.agg_center == 'FedAvg':
            global_weights = average_weights(local_weights)
            #global_weights = average_weights_unequal(local_weights, idx_num)        
        if args.agg_center == 'SFAT':
            idt_sorted = np.sort(idt)
            idtxnum = float('inf')
            idtx = args.topk
            if idtx > m:
                idtx = m
            if idtx != 0:
                idtxnum = idt_sorted[m-idtx]
            if epoch >0:
                global_weights = average_weights_alpha(local_weights, idt, idtxnum, args.pri)
                #global_weights = average_weights_alpha_unequal(local_weights, idt, idtxnum, args.pri, idx_num)                
            else:
                global_weights = average_weights(local_weights)
                #global_weights = average_weights_unequal(local_weights, idx_num)
            
        # update global weights
        mindspore.load_param_into_net(global_model, global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.set_train(False)
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], alg=args.agg_opt, anchor=global_model, anchor_mu=args.mu, local_rank=ipx)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            #idx_train_acc.append(idx_train)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            
        _, test_nat_acc = attack.eval_clean(global_model, test_loader, args.local_bs)
        _, test_pgd20_acc = attack.eval_robust(global_model, test_loader, args.local_bs, perturb_steps=20, epsilon=eps, step_size=sts,loss_fn="cent", category="Madry", random=True)
        
        if test_pgd20_acc >= global_best_pgd:
            global_best_pgd = test_pgd20_acc
            global_best_natural = test_nat_acc
            best_epoch = epoch

            save_checkpoint(global_model,checkpoint=args.out_dir,filename='bestpoint.ckpt')
        
        logger_test.append([args.epochs, args.local_ep, epoch, test_nat_acc, test_pgd20_acc])       
        print('Global Best Epoch: ', best_epoch)
        print('Global Nat Test Acc: {:.2f}%'.format(100*global_best_natural))
        print('Global PGD-20 Test Acc: {:.2f}%'.format(100*global_best_pgd))

    save_checkpoint(global_model,checkpoint=args.out_dir,filename='lastpoint.ckpt')

    # Test inference after completion of training
    logger_test.append([args.epochs, args.local_ep, best_epoch, global_best_natural, global_best_pgd])
    
    test_loss, test_nat_acc = attack.eval_clean(global_model, test_loader, args.local_bs)
    _, test_pgd20_acc = attack.eval_robust(global_model, test_loader, args.local_bs, perturb_steps=20, epsilon=eps, step_size=sts,loss_fn="cent", category="Madry", random=True)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))

    print('Nat Test Acc: {:.2f}%'.format(100*test_nat_acc))
    print('PGD-20 Test Acc: {:.2f}%'.format(100*test_pgd20_acc))
    # Saving the objects train_loss and train_accuracy:

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    logger_test.close()
