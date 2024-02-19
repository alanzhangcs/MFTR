import torch.nn as nn
import torch
import random

import sys
sys.path.append("../..")
from experiment_pvm.utils.mlp import MLP
from experiment_pvm.utils.attn import *
from experiment_pvm.utils.embed import *
from mobilev2 import *
class MLP_model(nn.Module):
    def __init__(self, gaze_encoder_out_dim, head_encoder_out_dim, pic_encoder_out_dim,dropout, seq_len, pred_len, pred_features, device):
        """ the fov prediction model, input is quad data, output is the predicted gaze points (x,y) sequences
        :param gaze_encoder_out_dim: the dimension of encoder output
        :param pred_features: the size of output feature, default is 2 (x, y)
        :param pred_len: the length of output sequence

        :return
        """
        super(MLP_model, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.device = device
        self.pred_feature = pred_features
        self.dropout = dropout
        self.head_embedding = nn.Linear(2, head_encoder_out_dim)
        self.gaze_embedding = nn.Linear(2, gaze_encoder_out_dim)
        self.head_rnn = nn.LSTM(
            input_size=head_encoder_out_dim,
            hidden_size=head_encoder_out_dim,
            num_layers=3,
            dropout=dropout,
            batch_first=True,
        )
        self.gaze_rnn = nn.LSTM(
            input_size=gaze_encoder_out_dim,
            hidden_size=gaze_encoder_out_dim,
            num_layers=3,
            dropout=dropout,
            batch_first=True,
        )
        self.Dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(gaze_encoder_out_dim + head_encoder_out_dim)

        temporal_attn_layers = 3

        self.temoral_attention = Attention([
                                    AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model=gaze_encoder_out_dim + head_encoder_out_dim,
                                                   n_heads=4) for i in range(temporal_attn_layers)
                                    ], self.norm)
        self.temporal_projection = nn.Linear(in_features=gaze_encoder_out_dim + head_encoder_out_dim, out_features=head_encoder_out_dim)
        self.head_prediction = MLP(head_encoder_out_dim, 2, dropout=self.dropout,
                             hidden_size=[128, 64])
        self.timeEmbeding = TimeEmbedding(d_model=gaze_encoder_out_dim + head_encoder_out_dim)
        self.token_type_embeddings = nn.Embedding(2, gaze_encoder_out_dim + head_encoder_out_dim)
        self.backbone = None
        self.finetune = nn.Linear(in_features=1000,out_features=pic_encoder_out_dim)
        self.spatial_time_embedding = TimeEmbedding(d_model=pic_encoder_out_dim)
        self.spatial_norm = nn.LayerNorm(pic_encoder_out_dim)
        self.spatial_attention = Attention([
                                    AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model=pic_encoder_out_dim,
                                                   n_heads=8) for i in range(temporal_attn_layers)
                                    ], self.spatial_norm)
        self.spatial_projection = nn.Linear(in_features=pic_encoder_out_dim, out_features=gaze_encoder_out_dim + head_encoder_out_dim)


        self.spatial_temporal_embedding = TimeEmbedding(d_model=gaze_encoder_out_dim + head_encoder_out_dim)
        self.modal_embedding = nn.Embedding(num_embeddings=2, embedding_dim=gaze_encoder_out_dim + head_encoder_out_dim)
        self.spatial_temporal_norm = nn.LayerNorm(gaze_encoder_out_dim + head_encoder_out_dim)
        self.spatial_temporal_attention = Attention([
                                    AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model=gaze_encoder_out_dim + head_encoder_out_dim,
                                                   n_heads=8) for i in range(temporal_attn_layers)
                                    ], self.spatial_temporal_norm)

        self.predictor = MLP(head_encoder_out_dim, pred_features, dropout=self.dropout,
                         hidden_size=[256,512])





    def forward(self, head, gaze, pic):
        """
        :param gaze:  batch x seq_len x 2
        :param head:  batch x seq_len x 2
        :param pic:   batch x (seq_len + pred_len) x w x h x 3
        """


        # gaze_feature/head_feature: batch x encoder_out_dim
        head_feature, (final_states_hn, final_states_cn) = self.head_rnn(self.head_embedding(head))
        gaze_feature, (final_states_hn, final_states_cn) = self.gaze_rnn(self.gaze_embedding(gaze))
        # feature = torch.cat((head_feature, gaze_feature), dim=2)
        moving_feature = torch.concat((gaze_feature, head_feature), dim=2)
        moving_feature = moving_feature + self.timeEmbeding(moving_feature)
        # moving_feature: batch x seq x （head_encoder_out_dim + tile_encoder_out_dim + gaze_encoder_out ）
        moving_feature = moving_feature + self.Dropout(self.temoral_attention(moving_feature,moving_feature,moving_feature))
        moving_feature = self.temporal_projection(moving_feature)

        head_results = self.head_prediction(moving_feature[:,-self.pred_len:,:])



        # image backbone
        pic = pic.permute(1,0,2,3,4)
        pic_featrues = []
        for i in range(self.pred_len+self.seq_len):

            feature = self.backbone(pic[i])
            feature  = self.finetune(feature)
            pic_featrues.append(feature)

        pic_featrues = torch.stack(pic_featrues)
        pic_featrues = pic_featrues.to(self.device)
        pic_featrues = pic_featrues.permute(1,0,2)


        pic_featrues = pic_featrues + self.spatial_time_embedding(pic_featrues)
        pic_featrues = pic_featrues + self.Dropout(self.spatial_attention(pic_featrues, pic_featrues, pic_featrues))
        pic_featrues = self.spatial_projection(pic_featrues)

        pic_featrues = pic_featrues + self.spatial_temporal_embedding(pic_featrues)
        pic_featrues = pic_featrues + self.modal_embedding(torch.full_like(torch.zeros((pic_featrues.shape[0], pic_featrues.shape[1])).to(self.device), 1, dtype=torch.int32))
        features = torch.concat((moving_feature, pic_featrues), dim=1)

        features = features + self.modal_embedding(torch.zeros_like(torch.zeros((moving_feature.shape[0], moving_feature.shape[1])).to(self.device), dtype=torch.int32))

        features = torch.concat((moving_feature, pic_featrues), dim=1)
        features = features + self.spatial_temporal_embedding(features)
        features = features + self.Dropout(self.spatial_temporal_attention(features, features, features))

        pred = self.predictor(features[:,-self.pred_len:,:])
        # return self.softmax(pred)
        return pred, head_results