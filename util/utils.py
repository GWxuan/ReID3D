import os
import sys
import time
import numpy as np
import pandas as pd
import collections
import random
import math
## For torch lib
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torch.nn.functional as F
## For Image lib
from PIL import Image
from plyfile import PlyData
import struct,time

def read_ply_velodyne(filename):
    ply_data = PlyData.read(filename)
    points = ply_data['vertex'].data.copy()
    cloud = np.empty([len(points), 3])
    for i in range(len(points)):
        point = points[i]
        p = np.array([point[0], point[1], point[2]])
        cloud[i] = p
    return np.array(cloud) - [np.mean(cloud[:,0]),np.mean(cloud[:,1]),np.min(cloud[:,2])]
    
def read_bin_velodyne(path):
    cloud = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])
    cloud = cloud[:,:3] 
    return cloud - [np.mean(cloud[:,0]),np.mean(cloud[:,1]),np.min(cloud[:,2])]

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
     
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # farthest = torch.randint(0, 1, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def process_labels(labels):
    unique_id = np.unique(labels)
    id_count = len(unique_id)
    id_dict = {ID:i for i, ID in enumerate(unique_id.tolist())}    
    for i in range(len(labels)):
        labels[i] = id_dict[labels[i]]
    assert len(unique_id)-1 == np.max(labels)
    return labels, id_count

class Video_train_Dataset(Dataset):
    def __init__(self, db_txt, info, seq_len=6, track_per_class=4, flip_p=0.5,
                 delete_one_cam=False, cam_type='cross_cam'):

        with open(db_txt, 'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))

        if delete_one_cam == True:
            info = np.load(info)
            info[:, 2], id_count = process_labels(info[:, 2])
            for i in range(id_count):
                idx = np.where(info[:, 2]==i)[0]
                if len(np.unique(info[idx, 3])) ==1:
                    info = np.delete(info, idx, axis=0)
                    id_count -=1
            info[:, 2], id_count = process_labels(info[:, 2])
        else:
            info = np.load(info)
            info[:, 2], id_count = process_labels(info[:, 2])

        self.info = []
        for i in range(len(info)):
            sample_clip = []
            F = info[i][1]-info[i][0]+1
            if F < seq_len:
                strip = list(range(info[i][0], info[i][1]+1))+[info[i][1]]*(seq_len-F)
                for s in range(seq_len):
                    pool = strip[s*1:(s+1)*1]
                    sample_clip.append(list(pool))
            else:
                interval = math.ceil(F/seq_len)
                strip = list(range(info[i][0], info[i][1]+1))+[info[i][1]]*(interval*seq_len-F)
                for s in range(seq_len):
                    pool = strip[s*interval:(s+1)*interval]
                    sample_clip.append(list(pool))
            self.info.append(np.array([np.array(sample_clip), info[i][2], info[i][3]], dtype=object))
            
        self.info = np.array(self.info)
        self.n_id = id_count
        self.n_tracklets = self.info.shape[0]
        self.flip_p = flip_p
        self.track_per_class = track_per_class
        self.cam_type = cam_type
        self.two_cam = False
        self.cross_cam = False
        self.num_points = 256

    def __getitem__(self, ID):
        sub_info = self.info[self.info[:, 1] == ID]

        if self.cam_type == 'normal':
            tracks_pool = list(np.random.choice(sub_info[:, 0], self.track_per_class))
        elif self.cam_type == 'two_cam':
            unique_cam = np.random.permutation(np.unique(sub_info[:, 2]))[:2]
            tracks_pool = list(np.random.choice(sub_info[sub_info[:, 2]==unique_cam[0], 0], 1))+\
                list(np.random.choice(sub_info[sub_info[:, 2]==unique_cam[1], 0], 1))
        elif self.cam_type == 'cross_cam':
            unique_cam = np.random.permutation(np.unique(sub_info[:, 2]))
            while len(unique_cam) < self.track_per_class:
                unique_cam = np.append(unique_cam, unique_cam)
            unique_cam = unique_cam[:self.track_per_class]
            tracks_pool = []
            for i in range(self.track_per_class):
                tracks_pool += list(np.random.choice(sub_info[sub_info[:, 2]==unique_cam[i], 0], 1))

        one_id_tracks = []

        for track_pool in tracks_pool:
            idx = np.random.choice(track_pool.shape[1], track_pool.shape[0])
            number = track_pool[np.arange(len(track_pool)), idx]
            imgs = []   
            for pc_path in self.imgs[number]:
                if pc_path[-3:] == 'bin':
                    pc = read_bin_velodyne(pc_path)
                else:
                    pc = read_ply_velodyne(pc_path)
                if len(pc) <= self.num_points:
                    random_idx = np.random.choice(len(pc),self.num_points-len(pc),replace=True)
                    pc = np.concatenate((pc, pc[random_idx]), axis=0)
                    pc = torch.tensor(pc)
                else:
                    random_idx = np.random.choice(len(pc),self.num_points,replace=False) 
                    random_idx.sort() 
                    pc = pc[random_idx]
                    pc = torch.tensor(pc)
                imgs.append(pc)
            imgs = torch.stack(imgs, dim=0)
            one_id_tracks.append(imgs)
        points_tracks = torch.stack(one_id_tracks, dim=0)
        return points_tracks,  ID*torch.ones(self.track_per_class, dtype=torch.int64)#, torch.stack(one_id_skel, dim=0)

    def __len__(self):
        return self.n_id

def Video_train_collate_fn(data):
    if isinstance(data[0], collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key:value  for key, value in zip(data[0].keys(), values)}
    else:
        imgs, labels= zip(*data)
        imgs = torch.cat(imgs, dim=0)
        labels = torch.cat(labels, dim=0)
        return imgs, labels

def Get_Video_train_DataLoader(db_txt, info, shuffle=True, num_workers=4, seq_len=10,
                               track_per_class=4, class_per_batch=8):
    dataset = Video_train_Dataset(db_txt, info, seq_len, track_per_class)
    dataloader = DataLoader(
        dataset, batch_size=class_per_batch, collate_fn=Video_train_collate_fn, shuffle=shuffle,
        worker_init_fn=lambda _:np.random.seed(), drop_last=True, num_workers=num_workers)
    return dataloader

class Video_test_rrs_Dataset(Dataset):
    def __init__(self, db_txt, info, query, seq_len=6, distractor=True):
        with open(db_txt, 'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))
        # info
        info = np.load(info)
        self.info = []
        for i in range(len(info)):
            if distractor == False and info[i][2]==0:
                continue
            sample_clip = []
            F = info[i][1]-info[i][0]+1
            if F < seq_len:
                strip = list(range(info[i][0], info[i][1]+1))+[info[i][1]]*(seq_len-F)
                for s in range(seq_len):
                    pool = strip[s*1:(s+1)*1]
                    sample_clip.append(list(pool))
            else:
                interval = math.ceil(F/seq_len)
                strip = list(range(info[i][0], info[i][1]+1))+[info[i][1]]*(interval*seq_len-F)
                for s in range(seq_len):
                    pool = strip[s*interval:(s+1)*interval]
                    sample_clip.append(list(pool))
            self.info.append(np.array([np.array(sample_clip), info[i][2], info[i][3]], dtype=object))

        self.info = np.array(self.info)
        self.n_id = len(np.unique(self.info[:, 1]))
        self.n_tracklets = self.info.shape[0]
        self.query_idx = np.load(query).reshape(-1)
        self.num_points = 256

        if distractor == False:
            zero = np.where(info[:, 2]==0)[0]
            self.new_query = []
            for i in self.query_idx:
                if i < zero[0]:
                    self.new_query.append(i)
                elif i <= zero[-1]:
                    continue
                elif i > zero[-1]:
                    self.new_query.append(i-len(zero))
                else:
                    continue
            self.query_idx = np.array(self.new_query)

    def __getitem__(self, idx):
        clips = self.info[idx, 0]
        imgs = []  
        skes = []          
        for pc_path in self.imgs[clips[:, 0]]:
            # time.sleep(0.01)
            if pc_path[-3:] == 'bin':
                pc = read_bin_velodyne(pc_path)
            else:
                pc = read_ply_velodyne(pc_path)
            if len(pc) == self.num_points:
                pc = torch.tensor(pc)
            elif len(pc) < self.num_points:
                random_idx = np.random.choice(len(pc),self.num_points-len(pc),replace=True)
                pc = np.concatenate((pc, pc[random_idx]), axis=0)
                pc = torch.tensor(pc)
            else:
                pc = torch.tensor(pc[np.newaxis,:,:])
                fps_idx = farthest_point_sample(pc, self.num_points)
                pc = pc[0][fps_idx[0]]
            imgs.append(pc)   
            skes.append(torch.tensor([0]))            

        skes = torch.stack(skes, dim=0)
        imgs = torch.stack(imgs, dim=0)
        label = self.info[idx, 1]*torch.ones(1, dtype=torch.int32)
        cam = self.info[idx, 2]*torch.ones(1, dtype=torch.int32)
        paths = [path for path in self.imgs[clips[:, 0]]]
        paths = np.stack(paths, axis=0)
        return imgs, label, cam, paths, skes
    def __len__(self):
        return len(self.info)

def Video_test_rrs_collate_fn(data):
    if isinstance(data[0], collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key:value  for key, value in zip(data[0].keys(), values)}
    else:
        imgs, label, cam, paths, skes= zip(*data)
        imgs = torch.cat(imgs, dim=0)
        labels = torch.cat(label, dim=0)
        cams = torch.cat(cam, dim=0)
        paths = np.concatenate(paths, axis=0)
        skes = torch.cat(skes, dim=0)
        return imgs, labels, cams, paths, skes

def Get_Video_test_rrs_DataLoader(db_txt, info, query, batch_size=10, shuffle=False,
                              num_workers=4, seq_len=6, distractor=True):
    dataset = Video_test_rrs_Dataset(db_txt, info, query, seq_len, distractor=distractor)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=Video_test_rrs_collate_fn, shuffle=shuffle,
        worker_init_fn=lambda _:np.random.seed(), num_workers=num_workers)
    return dataloader


