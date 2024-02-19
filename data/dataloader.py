'''data loader'''
import os
import numpy as np
import pickle
from torch.utils.data import Dataset
import torch
import imageio
from PIL import Image
from torchvision import transforms

class Dataset_fov(Dataset):

    def __init__(self, root_path, data_path, flag='train', seq_len=10, pred_len=5, require_pic=False):
        # init
        assert flag in ['train', 'test', 'valid']
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.require_pic = require_pic
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.__read_data__()

    def __read_data__(self):

        data_path_x = self.flag + '_X_quad_pic.pkl'
        data_path_y = self.flag + '_Y_quad_pic.pkl'

        x_path = os.path.join(self.root_path, self.data_path, data_path_x)
        y_path = os.path.join(self.root_path, self.data_path, data_path_y)

        x_dicts_pickle_file = open(x_path, 'rb')
        y_dicts_pickle_file = open(y_path, 'rb')

        x_dicts_pickle_file_1 = pickle.load(x_dicts_pickle_file)
        y_dicts_pickle_file_1 = pickle.load(y_dicts_pickle_file)
        self.data_x_quad = x_dicts_pickle_file_1['quad']
        self.data_y_quad = y_dicts_pickle_file_1['quad']
        self.data_x_pic = x_dicts_pickle_file_1['pic']
        self.data_y_pic = y_dicts_pickle_file_1['pic']


        x_dicts_pickle_file.close()
        y_dicts_pickle_file.close()

    def __getitem__(self, idx):
        #head_move
        data = self.data_x_quad[idx].astype(np.float32)

        data[:,0:1] =  (90 - self.data_x_quad[idx].astype(np.float32)[:,0:1]) / 180
        data[:,1:2] =  (180 - self.data_x_quad[idx].astype(np.float32)[:,1:2]) / 360

        data_y = self.data_y_quad[idx].astype(np.float32)
        data_y[:,0:1] = (90 - self.data_y_quad[idx].astype(np.float32)[:,0:1]) / 180
        data_y[:, 1:2] = (180 - self.data_y_quad[idx].astype(np.float32)[:, 1:2]) / 360



        if self.require_pic:
            pic_data = []
            pic_path_set = self.data_x_pic[idx]
            for pic_path in pic_path_set:
                input_image = Image.open(pic_path)

                input_tensor = input_image
                pic_data.append(input_tensor)

            pic_data_set = self.data_y_pic[idx]

            for i in range(self.pred_len):
                pic_path = pic_data_set[i]
                input_image = Image.open(pic_path)
                input_tensor = input_image
                pic_data.append(input_tensor)
            pic_data = torch.stack(pic_data)




            return data[:,3:], data[:,0:2], data_y[:self.pred_len,0:2], pic_data
        else:

            return data[:,3:], data[:,0:2], data_y[:self.pred_len,0:2]

    def __len__(self):
        assert len(self.data_x_quad) == len(self.data_y_quad)
        return len(self.data_x_quad)

