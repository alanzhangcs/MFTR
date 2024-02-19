import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torchvision
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
from mobilev2 import *
import sys
sys.path.append("../..")
from model_1 import MLP_model
from experiment_pvm.utils.tools import quad_to_label, calcute_overlap, get_accuracy
from experiment_pvm.utils.dataloader import Dataset_fov
from experiment_pvm.utils.earlyStop import EarlyStopping
from fvcore.nn import FlopCountAnalysis, parameter_count_table
class Exp_LSTM(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self.build_model().to(self.device)



    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            # print(torch.cuda.device_count())
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def build_model(self):
        """

        :return:
        """
        model = MLP_model(gaze_encoder_out_dim=self.args.gaze_encoder_out_dim, head_encoder_out_dim=self.args.head_encoder_out_dim, pic_encoder_out_dim=self.args.pic_encoder_out_dim, dropout=self.args.dropout,
                          seq_len=self.args.seq_len,
                          pred_len=self.args.pred_len, pred_features=self.args.tile_num_x * self.args.tile_num_y, device=self.device)
        model.apply(initialize_parameters)
        # model.backbone = MobileNetV2(num_classes=self.args.pic_encoder_out_dim)
        model.backbone = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
        return model

    def get_data(self, flag):
        """

        :param flag:
        :return:
        """

        data_path = self.args.data_path
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = self.args.batch_size
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            # Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = self.args.batch_size

        data_set = Dataset_fov(
            root_path=self.args.root_path,
            data_path=data_path,
            flag=flag,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            require_pic=True
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last)

        return None, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def _select_mse_criterion(self):
        # criterion = Tile_Loss(tile_num_x=tile_num_x, tile_num_y=tile_num_y)
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion, criterion_mse):
        self.model.eval()
        total_loss = []
        total_accuracy = []
        total_overlap = []
        for i, (batch_x_quad, batch_x_head, batch_y_quad, batch_pic) in enumerate(vali_loader):
            pred, pos, label, id = self.process_one_batch_with_type(batch_x_quad, batch_x_head, batch_y_quad[:,:self.args.pred_len,:],batch_pic)
            accuracy, overlap = get_accuracy(pred, id)

            pred = pred.reshape(-1, self.args.tile_num_y * self.args.tile_num_x)
            label = label.reshape(-1, self.args.tile_num_y * self.args.tile_num_x)

            loss2 = criterion(pred, label)
            loss_mse = criterion_mse(pos, batch_y_quad[:, :self.args.pred_len, :].to(self.device))
            loss = self.args.alpha * loss_mse + self.args.belta * loss2
            total_loss.append(loss.item())
            total_accuracy.append(accuracy)
            total_overlap.append(overlap)
        total_loss = np.average(total_loss)
        total_accuracy = torch.stack(total_accuracy)
        total_accuracy = torch.mean(total_accuracy)

        total_overlap = torch.stack(total_overlap)
        total_overlap = torch.mean(total_overlap)
        self.model.train()
        return total_loss, total_accuracy, total_overlap



    def process_one_batch_with_type(self, batch_gaze, batch_head,  batch_y_quad, batch_pic, type="train"):
        batch_gaze = batch_gaze.float().to(self.device)
        batch_head = batch_head.float().to(self.device)
        batch_pic = batch_pic.float().to(self.device)
        batch_y_quad = batch_y_quad.float().to(self.device)
        torch.cuda.synchronize()
        outputs, pos = self.model.forward(batch_head, batch_gaze, batch_pic)

        # flops = FlopCountAnalysis(self.model, (batch_head, batch_gaze, batch_pic))
        # print('FLOPs = ' + str(flops.total() / 1000 ** 3) + 'G')

        batch_label, batch_tile_id = quad_to_label(batch_y_quad, self.args.tile_num_x, self.args.tile_num_y)

        return outputs.to(self.device), pos.to(self.device), batch_label.to(self.device), batch_tile_id.to(self.device)

    def train(self, setting):
        train_data, train_loader = self.get_data(flag='test')
        vali_data, vali_loader = self.get_data(flag='valid')
        test_data, test_loader = self.get_data(flag='test')

        # model的存放路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # log路径用于存放loss值
        writer_path = os.path.join(self.args.checkpoints, 'log')
        # print("writer_path:", writer_path)
        if not os.path.exists(writer_path):
            os.mkdir(writer_path)
        writer = SummaryWriter(writer_path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        mse_cir = self._select_mse_criterion()

        f = open('log/seq_len_{}_pred_len_{}.txt'.format(self.args.seq_len, self.args.pred_len,), 'w+')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            final_loss = []
            train_accuracy = []
            train_overlap = []
            train_mse_loss = []
            self.model.train()

            for batch_x_quad, batch_x_head, batch_y_quad, batch_pic, in tqdm(train_loader):
                # print(batch_x.size())
                # print(batch_y.size())
                iter_count += 1
                model_optim.zero_grad()
                pred, pos, label, id = self.process_one_batch_with_type(batch_x_quad, batch_x_head, batch_y_quad, batch_pic)
                # print("pred")
                # print(pred.shape)
                # print("id")
                # print(id.shape)
                accuracy, overlap = get_accuracy(pred, id)
                # pred: (batch * pred_len )x ( tile_num)

                pred = pred.reshape(-1, self.args.tile_num_y * self.args.tile_num_x)
                label = label.reshape(-1, self.args.tile_num_y * self.args.tile_num_x)

                # loss = torch.sqrt(criterion(pred, true))
                loss2 = criterion(pred, label)
                mse_loss = mse_cir(pos, batch_y_quad[:, :self.args.pred_len, :].to(self.device))
                train_loss.append(loss2.item())
                train_accuracy.append(accuracy)
                train_overlap.append(overlap)
                loss = self.args.alpha * mse_loss + self.args.belta * loss2
                final_loss.append(loss.item())
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            final_loss = np.average(final_loss)
            train_mse_loss = np.average(train_mse_loss)
            train_accuracy = torch.stack(train_accuracy)
            train_accuracy = torch.mean(train_accuracy)
            train_overlap = torch.stack(train_overlap)
            train_overlap = torch.mean(train_overlap)
            vali_loss, vali_accuracy, vali_overlap = self.vali(vali_loader, criterion, mse_cir)
            test_loss, test_accuracy, test_overlap = self.vali(test_loader, criterion, mse_cir)

            # self.args.in_channels, self.args.out_channels, self.args.kernel_size, self.args.stride, self.args.dropout,
            # self.args.activation, self.args.padding, self.args.in_features, self.args.out_features, self.args.n_heads,
            # self.args.num_layers, self.args.d_model, self.args.d_ff, self.args.d_layers,
            # self.args.quad_features, self.args.encoder_in_quad_features, self.args.decoder_in_quad_features,
            # self.args.pred_len, self.args.label_len, self.args.pred_features, self.args.devices

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, final_loss, vali_loss, test_loss))
            f.write("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}\n".format(
                epoch + 1, train_steps, final_loss, vali_loss, test_loss))
            print("Train overlap: {0:.7f} Vali overlap: {1:.7f} Test overlap: {2:.7f}".format(train_overlap,
                                                                                              vali_overlap,
                                                                                              test_overlap))
            print( "Train Accuracy: {0:.7f} Vali accuracy: {1:.7f} Test accuracy: {2:.7f}".format(train_accuracy, vali_accuracy, test_accuracy))
            f.write("Train Accuracy: {0:.7f} Vali accuracy: {1:.7f} Test accuracy: {2:.7f}\n".format(train_accuracy, vali_accuracy, test_accuracy))
            f.flush()
            writer.add_scalars(
                '{}_seq_len={}_pred_len={}_encoder_input={}_decoder_input={}'
                .format(self.args.model,self.args.seq_len, self.args.pred_len, self.args.encoder_input_size, self.args.decoder_input_size
                        ),
                {'train': train_loss, 'valid': vali_loss, 'test_loss': test_loss},
                global_step=epoch)

            early_stopping(-vali_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        f.close()



def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)


parser = argparse.ArgumentParser(description='fov prediction lstm')

parser.add_argument('--model', type=str,  default='cls_gaze_only_mlp',help='experiment model')
parser.add_argument('--root_path', type=str, default='/home/kb/PVS-HMEM_database_final/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='Generated_data', help='data file dir')
parser.add_argument('--checkpoints', type=str, default='/data/data_kb/360/checkpoints/cls_gaze_only_mlp', help='location of model_lstm checkpoints')
parser.add_argument('--activation', type=str, default='relu',help='activation')
parser.add_argument('--seq_len', type=int, default=5, help='input sequence length of Informer encoder')
parser.add_argument('--gaze_encoder_out_dim', type=int, default=256, help='')
parser.add_argument('--head_encoder_out_dim', type=int, default=256, help='')
parser.add_argument('--pic_encoder_out_dim', type=int, default=100, help='')
# parser.add_argument('--num_layers', type=int, default=1, help='num layers of lstm model')
# parser.add_argument('--hidden_size', type=int, default=128, help='hidden size of lstm model')
parser.add_argument('--quad_feature', type=int, default=2, help='encoder/decoder quad input size')
parser.add_argument('--encoder_input_size', type=int, default=128, help='input size for lstm encoder')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length')
parser.add_argument('--decoder_input_size', type=int, default=256, help='input size for lstm decoder')
parser.add_argument('--decoder_output_size', type=int, default=256, help='output size for lstm decoder')
parser.add_argument('--pred_features', type=int, default=10*20 , help='output size')
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
parser.add_argument('--num_workers', type=int, default=12, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=1000, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=40, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=2, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--tile_num_x', type=int, default=10, help='360 video split into tile_num_x')
parser.add_argument('--tile_num_y', type=int, default=20, help='360 video split into tile_num_y')
parser.add_argument('--fov_size', type=int, default=5, help='360 video split into tile_num_y')
parser.add_argument('--alpha', type=float, default=0.35, help='parameter for mse loss')
parser.add_argument('--belta', type=float, default=0.65, help='parameter for cross loss')
args = parser.parse_args()

lstm_exp = Exp_LSTM(args)
setting = '{}_seq_len={}_pred_len={}_encoder_input={}_decoder_input={}'.format(args.model, args.seq_len, args.pred_len, args.encoder_input_size,
                                                                               args.decoder_input_size)
print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
lstm_exp.train(setting)
