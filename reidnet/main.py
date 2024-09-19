import parser
args = parser.parse_args()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import math
import random
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark=True
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

import sys
sys.path.append('..')
from model import loss
from model import network
from util import utils

def np_cdist(feat1, feat2):
    """Cosine distance"""
    feat1_u = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True) # n * d -> n
    feat2_u = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True) # n * d -> n
    return -1 * np.dot(feat1_u, feat2_u.T)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def Video_acc(features, ids, q_idx):
    
    features = torch.tensor(features)
    q_idx = list(q_idx)
    g_idx = [i for i in range(len(ids)) if i not in q_idx]
    # import pdb;pdb.set_trace()
    fea_g = features[g_idx]
    fea_q = features[q_idx]
    ids_g = ids[g_idx]
    ids_q = ids[q_idx]
    acc_right = 0
    acc_right3 = 0
    
    right_ids = []
    ap = 0
    for query in q_idx:
        fea_dis = torch.tensor(np_cdist(fea_g, features[query].reshape(1,-1)).reshape(-1))
        min_idx = torch.argmin(fea_dis)
        a1, idx1 = torch.sort(fea_dis, descending=False)
        if ids_g[min_idx] == ids[query]:
            acc_right += 1
            right_ids.append(int(ids_g[min_idx]))
        if ids[query] in ids_g[idx1[:3]]:
            acc_right3 += 1
        
        good_index = [k for k in range(len(ids_g)) if ids_g[k] == ids[query]]
        ngood = len(good_index)
        mask = np.in1d(idx1.numpy(), good_index)
        rows_good = np.argwhere(mask==True)
        rows_good = rows_good.flatten()   
        for i in range(ngood):
            d_recall = 1.0/ngood
            precision = (i+1)*1.0/(rows_good[i]+1)
            ap = ap + d_recall*precision
        
    print('acc: ', acc_right/len(q_idx), acc_right3/len(q_idx), ap/len(q_idx), acc_right, len(q_idx))
    return acc_right/len(q_idx), acc_right3/len(q_idx),  ap/len(q_idx), acc_right, len(q_idx)    
      
def test_rrs(net, dataloader, args):

    net.eval()
    pbar = tqdm(total=len(dataloader), ncols=100, leave=True)
    pbar.set_description('Inference')

    gallery_features = []
    gallery_features_ske = []
    gallery_labels = []
    gallery_cams = []
    with torch.no_grad():
        for c, data in enumerate(dataloader):
            seqs = data[0].cuda()
            seqs = seqs.reshape((seqs.shape[0]//args.seq_len, args.seq_len, ) + seqs.shape[1:])
            label = data[1]
            cams = data[2]
            out = net(seqs)
            feat = out['val_bn']
            gallery_features.append(feat.cpu())
            gallery_labels.append(label)
            gallery_cams.append(cams)
            pbar.update(1)
    pbar.close()
    gallery_features = torch.cat(gallery_features, dim=0).numpy()
    gallery_labels = torch.cat(gallery_labels, dim=0).numpy()
    gallery_cams = torch.cat(gallery_cams, dim=0).numpy()
        
    acc = Video_acc(gallery_features, gallery_labels, dataloader.dataset.query_idx)
    net.train()
    return acc

if __name__ == '__main__':
    torch.manual_seed(4)
    np.random.seed(4)
    random.seed(4)
    num=4
    torch.set_num_threads(num)

    print('\nDataloading starts !!')

    train_dataloader = utils.Get_Video_train_DataLoader(
        args.train_txt, args.train_info, shuffle=True,num_workers=args.num_workers,
        seq_len=args.seq_len, track_per_class=args.track_per_class, class_per_batch=args.class_per_batch)

    test_rrs_dataloader = utils.Get_Video_test_rrs_DataLoader(
        args.test_txt, args.test_info, args.query_info, batch_size=args.test_batch,
        shuffle=False, num_workers=args.num_workers, seq_len=args.seq_len, distractor=True)

    print('Dataloading ends !!\n')

    num_class = train_dataloader.dataset.n_id
    net = nn.DataParallel(
        network.reid3d(args.feat_dim, num_class=num_class, stride=args.stride).cuda())

    if args.load_ckpt is not None:
        state = torch.load(args.load_ckpt)
        net.load_state_dict(state,strict=False)
        
    # initialize_weights(net)
    os.system('mkdir -p %s'%(args.ckpt))

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum=0.9, weight_decay = 1e-4)
    else:
        optimizer = optim.AdamW(net.parameters(), lr = args.lr, betas=(0.9,0.99), weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=args.lr*0.01, last_epoch=-1, verbose=False)

    best_cmc = 0
    loss = loss.Loss().cuda()
    for epoch in range(0, args.n_epochs):

        ############################### Training ###############################
        pbar = tqdm(total=len(train_dataloader), ncols=100, leave=True)
        pbar.set_description('Epoch %03d' %epoch)
        
        loss_all = 0
        loss_id = 0
        loss_trip = 0
        loss_id_final = 0
        loss_trip_final = 0
        for batch_idx, data in enumerate(train_dataloader):
            seqs, labels = data
            num_batches = seqs.size()[0]
            seqs = seqs.cuda()
            labels = labels.cuda()
            out = net(seqs)
            loss_out = loss(out, labels)

            total_loss = 0
            total_loss += loss_out['track_id']
            total_loss += loss_out['trip']
            
            loss_all += float(total_loss)
            loss_id += float(loss_out['track_id'])
            loss_trip += float(loss_out['trip'])
            
            loss_id_final += float(loss_out['track_id_final'])
            loss_trip_final += float(loss_out['trip_final'])
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            pbar.update(1)
        pbar.close()
        print('lr: ', optimizer.state_dict()['param_groups'][0]['lr'], 'total_loss: ',loss_all, 'track_id_loss: ', loss_id, 'trip_loss: ', loss_trip, 
                'track_id_final: ', loss_id_final, 'trip_final: ', loss_trip_final)
        torch.save(net.state_dict(), os.path.join(args.ckpt, 'ckpt_last.pth'))
        if args.lr_step_size !=0:
            scheduler.step()
        f = open(os.path.join(args.ckpt, args.log_path), 'a')
        f.write('lr: %f total loss: %.4f track_id loss: %.4f trip loss: %.4f track_id fianl: %.4f trip final: %.4f \n'%
                (optimizer.state_dict()['param_groups'][0]['lr'], loss_all, loss_id, loss_trip, loss_id_final, loss_trip_final))
        f.close()

        ############################### Validation ###############################
        if (epoch+1) % args.eval_freq == 0:
            acc, acc3, map, num_right, num_all = test_rrs(net, test_rrs_dataloader, args)

            f = open(os.path.join(args.ckpt, args.log_path), 'a')
            f.write('[Epoch %03d] top1: %.1f top3: %.1f map: %.1f num_right: %d num_all: %d \n'%(epoch, acc*100, acc3*100, map*100, num_right, num_all))

            if acc >= best_cmc:
                torch.save(net.state_dict(), os.path.join(args.ckpt, 'ckpt_best.pth'))
                best_cmc = acc

            f.close()

    print('best_acc: ', best_cmc)
    f = open(os.path.join(args.ckpt, args.log_path), 'a')
    f.write('best acc: %.1f \n'%(best_cmc*100))
    f.close()


############################### Test ###############################
    print('----Test---- \n')
    acc, acc3, map, num_right, num_all = test_rrs(net, test_rrs_dataloader, args)

    f = open(os.path.join(args.ckpt, args.log_path), 'a')
    f.write('[Test] top1: %.1f top3: %.1f map: %.1f num_right: %d num_all: %d \n'%(acc*100, acc3*100, map*100, num_right, num_all))
    f.close()