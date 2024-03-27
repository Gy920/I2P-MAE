import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
import open3d as o3d

@DATASETS.register_module()
class Masstar(data.Dataset):
    def __init__(self, config):
        # self.data_list_file = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.npoints = config.N_POINTS
        self.filelist = []
        self.key = []
        self.subset=config.subset
        
        
        self.sample_points_num = config.npoints
        self.data_list_file = os.path.join(config.DATA_PATH, f'{self.subset}.txt')

        with open(self.data_list_file,'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line)
                line = f.readline()

        self.gt_path = self.pc_path
        for key in self.filelist:
            # print(key)
            self.key.append(key)


        # self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < num:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:num]]
        
    def __getitem__(self, idx):
        key = self.key[idx].replace('\n','')
        pc_path=os.path.join(self.pc_path,key.split('/')[0]+'/output.ply')
        point_cloud = o3d.io.read_point_cloud(pc_path)
        pc= np.asarray(point_cloud.points).astype(np.float32)
        pc=self.random_sample(pc,self.sample_points_num)
        pc = self.pc_norm(pc)

        return key, key, torch.asarray(pc)

    def __len__(self):
        return len(self.key)