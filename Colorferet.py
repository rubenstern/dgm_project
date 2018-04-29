
# coding: utf-8


import numpy as np
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import random
import os
from scipy import misc
from skimage.transform import resize



class Colorferet(object):

    def __init__(self, train=True):
        self.data_root = 'data_uncompressed/'
        missing_folder = [11,226,232,240,274,277,372,382,463,464,649,651,696,702]
        
        if train:
            self.train = True
            self.persons = [x for x in range(1,650) if x not in missing_folder]
        else:
            self.train = False
            self.persons = [x for x in range(650,740) if x not in missing_folder]

        self.data = {}
        self.len = 0
        self.folder_names = list(map(lambda x: ('00000'+ str(x))[-5:], self.persons))
        self.cpt = 0
        for folder_name in self.folder_names:
            data_person = []
            full_folder_name = 'data_uncompressed/' + folder_name + '/'
            for file_name in os.listdir(full_folder_name):
                im = misc.imread(full_folder_name + file_name)/255.0 # shape (768, 512, 3) 
                im = resize(im, (128,128,3)) # shape (128, 128, 3)
                data_person.append(im) 
                self.len += 1
            self.data[self.cpt] = np.array(data_person)
            self.cpt += 1
     


    # to speed up training of drnet, don't get a whole sequence when we only need 4 frames
    # x_c1, x_c2, x_p1, x_p2 
    def get_drnet_data(self):
        person_idx = np.random.randint(self.cpt)
        person_data = self.data[person_idx]
        seq_len = len(person_data)
           
        seq = [] 
        for i in range(4):
            t = np.random.randint(seq_len)
            im = person_data[t]
            seq.append(im)
        return np.array(seq)

    def __getitem__(self, index):
        random.seed(index)
        np.random.seed(index)
        return torch.from_numpy(self.get_drnet_data())

    def __len__(self):
        return self.len



