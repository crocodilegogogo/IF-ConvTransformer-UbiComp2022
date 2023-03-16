import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import time
from utils.utils import *
import os
from torch.nn.utils import weight_norm
from contiguous_params import ContiguousParams

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.transpose(1,2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        # x = x + Variable(self.pe, requires_grad=False)
        return self.dropout(x)

class SelfAttention(nn.Module):
    def __init__(self, k, heads = 8, drop_rate = 0):
        super(SelfAttention, self).__init__()
        self.k, self.heads = k, heads
        # input 维度为k（embedding结果），map成一个k*heads维度的矩阵
        self.tokeys    = nn.Linear(k, k * heads, bias = False)
        self.toqueries = nn.Linear(k, k * heads, bias = False)
        self.tovalues  = nn.Linear(k, k * heads, bias = False)
        # 设置dropout
        self.dropout_attention = nn.Dropout(drop_rate)
        # 在通过线性转换把维度压缩到 k
        self.unifyheads = nn.Linear(heads * k, k)
        
    def forward(self, x):
        
        b, t, k = x.size()
        h = self.heads
        queries = self.toqueries(x).view(b, t, h, k)
        keys    = self.tokeys(x).view(b, t, h, k)
        values  = self.tovalues(x).view(b, t, h, k)
        # 把 head 压缩进 batch的dimension
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys    = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values  = values.transpose(1, 2).contiguous().view(b * h, t, k)
        # 这等效于对点积进行normalize
        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))
        # 矩阵相乘
        dot  = torch.bmm(queries, keys.transpose(1,2))
        # 进行softmax归一化
        dot = F.softmax(dot, dim=2)
        dot = self.dropout_attention(dot)
        out = torch.bmm(dot, values).view(b, h, t, k)
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h*k)
        
        return self.unifyheads(out) # (b, t, k)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads, drop_rate):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(k, heads = heads, drop_rate = drop_rate)
        self.norm1 = nn.LayerNorm(k)

        self.mlp = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )
        self.norm2 = nn.LayerNorm(k)
        self.dropout_forward = nn.Dropout(drop_rate)

    def forward(self, x):
        
        # 先做self-attention
        attended = self.attention(x)
        # 再做layer norm
        x = self.norm1(attended + x)
        # feedforward和layer norm
        feedforward = self.mlp(x)
        
        return self.dropout_forward(self.norm2(feedforward + x))

class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()
    
class IMU_Fusion_Block(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, 
                 feature_channel, kernel_size_grav,
                 scale_num, dataset_name):
        super(IMU_Fusion_Block, self).__init__()
        
        self.scale_num         = scale_num
        self.input_channel     = input_channel
        self.tcn_grav_convs    = []
        self.tcn_gyro_convs    = []
        self.tcn_acc_convs     = []
        
        for i in range(self.scale_num):
            
            dilation_num_grav = i+1
#            padding_grav = (kernel_size_grav - 1) * dilation_num_grav // 2
#            kernel_size_gyro = padding_grav*2-1
#            kernel_size_acc = padding_grav*2+1
            padding_grav     = (kernel_size_grav - 1) * dilation_num_grav
            kernel_size_gyro = padding_grav
            kernel_size_acc  = padding_grav + 1
            
            tcn_grav = nn.Sequential(
                weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                      (1,kernel_size_grav), 1, (0,padding_grav), 
                                      dilation=dilation_num_grav)),
                # nn.BatchNorm2d(feature_channel),
                Chomp2d(padding_grav),
                nn.ReLU(),
                # nn.MaxPool2d(2)
                )
            
            
            if kernel_size_gyro == 1:
                tcn_gyro = nn.Sequential(
                    weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                          (1,1), 1, (0,0), 
                                          dilation=1)),
                    # nn.BatchNorm2d(feature_channel),
                    nn.ReLU(),
                    # nn.MaxPool2d(2)
                    )
            else:
                tcn_gyro = nn.Sequential(
                    weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                          (1,kernel_size_gyro), 1, (0,(kernel_size_gyro-1)*1), 
                                          dilation=1)),
                    # nn.BatchNorm2d(feature_channel),
                    Chomp2d((kernel_size_gyro-1)*1),
                    nn.ReLU(),
                    # nn.MaxPool2d(2)
                    )
            
            tcn_acc = nn.Sequential(
                weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                      (1,kernel_size_acc), 1, (0,(kernel_size_acc-1)*1), 
                                      dilation=1)),
                # nn.BatchNorm2d(feature_channel),
                Chomp2d((kernel_size_acc-1)*1),
                nn.ReLU(),
                # nn.MaxPool2d(2)
                )
            
            setattr(self, 'tcn_grav_convs%i' % i, tcn_grav)
            self.tcn_grav_convs.append(tcn_grav)
            setattr(self, 'tcn_gyro_convs%i' % i, tcn_gyro)
            self.tcn_gyro_convs.append(tcn_gyro)
            setattr(self, 'tcn_acc_convs%i' % i, tcn_acc)
            self.tcn_acc_convs.append(tcn_acc)
        
        self.attention = nn.Sequential(
                nn.Linear(3*feature_channel, 1),
                nn.Tanh()
                )
        
        self.attention_scale = nn.Sequential(
                nn.Linear(3*feature_channel, 1),
                nn.Tanh()
                )
        
    def forward(self, x):
        
        x_grav = x[:,:,0:3,:]
        x_gyro = x[:,:,3:6,:]
        x_acc  = x[:,:,6:9,:]
        
        for i in range(self.scale_num):
            
            out_grav = self.tcn_grav_convs[i](x_grav).unsqueeze(4)
            out_gyro = self.tcn_gyro_convs[i](x_gyro).unsqueeze(4)
            out_acc  = self.tcn_acc_convs[i](x_acc).unsqueeze(4)
            
            if i == 0:
                out_attitude = torch.cat([out_grav, out_gyro], dim=4)
                out_dynamic  = out_acc
            else:
                out_attitude = torch.cat([out_attitude, out_grav], dim=4)
                out_attitude = torch.cat([out_attitude, out_gyro], dim=4)
                out_dynamic  = torch.cat([out_dynamic, out_acc], dim=4)
                # out_attitude = out_attitude + (out_grav + out_gyro)
                # out_dynamic  = out_dynamic + out_acc
        
        # (batch_size, time_length, sensor_num*scale_num, 3(xyz), feature_chnnl)
        out_attitude = out_attitude.permute(0,3,4,2,1)
        # (batch_size, time_length, sensor_num*scale_num, 3(xyz)*feature_chnnl)
        out_attitude = out_attitude.reshape(out_attitude.shape[0], out_attitude.shape[1], out_attitude.shape[2], -1)
        # time-step-wise sensor attention, sensor_attn:(batch_size, time_length, sensor_num*scale_num, 1)
        sensor_attn  = self.attention(out_attitude).squeeze(3)
        sensor_attn  = F.softmax(sensor_attn, dim=2).unsqueeze(-1)
        out_attitude = sensor_attn * out_attitude
        # (batch_size, time_length, sensor_num*scale_num, 3(xyz), feature_chnnl)
        out_attitude = out_attitude.reshape(out_attitude.shape[0], out_attitude.shape[1], out_attitude.shape[2], 3, -1)
        # (batch_size, time_length, sensor_num*scale_num*3(xyz), feature_chnnl)
        out_attitude = out_attitude.reshape(out_attitude.shape[0], out_attitude.shape[1], out_attitude.shape[2]*3, -1)
        # (batch_size, feature_chnnl, sensor_num*scale_num*3(xyz), time_length)
        out_attitude = out_attitude.permute(0,3,2,1)
        
#        # plus all the different scales
#        out_attitude = torch.split(out_attitude, 3, dim=2)
#        for j in range(len(out_attitude)):
#            if j == 0:
#                sum_attitude = out_attitude[j]
#            else:
#                sum_attitude = sum_attitude + out_attitude[j]
#        out_attitude = sum_attitude
        
        # (batch_size, time_length, scale_num, 3(xyz), feature_chnnl)
        out_dynamic = out_dynamic.permute(0,3,4,2,1)
        # (batch_size, time_length, scale_num, 3(xyz)*feature_chnnl)
        out_dynamic = out_dynamic.reshape(out_dynamic.shape[0], out_dynamic.shape[1], out_dynamic.shape[2], -1)
        # time-step-wise sensor attention, scale_attn:(batch_size, time_length, scale_num, 1)
        scale_attn  = self.attention_scale(out_dynamic).squeeze(3)
        scale_attn  = F.softmax(scale_attn, dim=2).unsqueeze(-1)
        out_dynamic = scale_attn * out_dynamic
        # (batch_size, time_length, scale_num, 3(xyz), feature_chnnl)
        out_dynamic = out_dynamic.reshape(out_dynamic.shape[0], out_dynamic.shape[1], out_dynamic.shape[2], 3, -1)
        # (batch_size, time_length, scale_num*3(xyz), feature_chnnl)
        out_dynamic = out_dynamic.reshape(out_dynamic.shape[0], out_dynamic.shape[1], out_dynamic.shape[2]*3, -1)
        # (batch_size, feature_chnnl, scale_num*3(xyz), time_length)
        out_dynamic = out_dynamic.permute(0,3,2,1)
        
        # concatenate all the different scales
        out_attitude = torch.split(out_attitude, 6, dim=2)
        for j in range(len(out_attitude)):
            per_scale_attitude = torch.split(out_attitude[j], 3, dim=2)
            for k in range(len(per_scale_attitude)):
                if k == 0:
                    per_attitude   = per_scale_attitude[k]
                else:
                    per_attitude   = per_attitude + per_scale_attitude[k]
            if j == 0:
                all_attitude = per_attitude
            else:
                all_attitude = torch.cat([all_attitude, per_attitude], dim=2)
        out_attitude = all_attitude
        
        out          = torch.cat([out_attitude, out_dynamic], dim = 2)
        
        return out, sensor_attn
    
class IMU_Fusion_Block_With_Mag(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, 
                 feature_channel, kernel_size_grav,
                 scale_num, dataset_name):
        super(IMU_Fusion_Block_With_Mag, self).__init__()
        
        self.scale_num         = scale_num
        self.input_channel     = input_channel
        self.tcn_grav_convs    = []
        self.tcn_mag_convs     = []
        self.tcn_gyro_convs    = []
        self.tcn_acc_convs     = []
        
        for i in range(self.scale_num):
            
            dilation_num_grav = i+1
#            padding_grav = (kernel_size_grav - 1) * dilation_num_grav // 2
#            kernel_size_gyro = padding_grav*2-1
#            kernel_size_acc = padding_grav*2+1
            padding_grav     = (kernel_size_grav - 1) * dilation_num_grav
            kernel_size_gyro = padding_grav
            kernel_size_acc  = padding_grav + 1
            
            tcn_grav = nn.Sequential(
                weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                      (1,kernel_size_grav), 1, (0,padding_grav), 
                                      dilation=dilation_num_grav)),
                # nn.BatchNorm2d(feature_channel),
                Chomp2d(padding_grav),
                nn.ReLU(),
                # nn.MaxPool2d(2)
                )
            
            tcn_mag = nn.Sequential(
            # the mag kernel params are the same with tcn_grav
            weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                  (1,kernel_size_grav), 1, (0,padding_grav), 
                                  dilation=dilation_num_grav)),
            # nn.BatchNorm2d(feature_channel),
            Chomp2d(padding_grav),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
            
            if kernel_size_gyro == 1:
                tcn_gyro = nn.Sequential(
                    weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                          (1,1), 1, (0,0), 
                                          dilation=1)),
                    # nn.BatchNorm2d(feature_channel),
                    nn.ReLU(),
                    # nn.MaxPool2d(2)
                    )
            else:
                tcn_gyro = nn.Sequential(
                    weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                          (1,kernel_size_gyro), 1, (0,(kernel_size_gyro-1)*1), 
                                          dilation=1)),
                    # nn.BatchNorm2d(feature_channel),
                    Chomp2d((kernel_size_gyro-1)*1),
                    nn.ReLU(),
                    # nn.MaxPool2d(2)
                    )
            
            tcn_acc = nn.Sequential(
                weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                      (1,kernel_size_acc), 1, (0,(kernel_size_acc-1)*1), 
                                      dilation=1)),
                # nn.BatchNorm2d(feature_channel),
                Chomp2d((kernel_size_acc-1)*1),
                nn.ReLU(),
                # nn.MaxPool2d(2)
                )
            
            setattr(self, 'tcn_grav_convs%i' % i, tcn_grav)
            self.tcn_grav_convs.append(tcn_grav)
            setattr(self, 'tcn_mag_convs%i' % i, tcn_mag)
            self.tcn_mag_convs.append(tcn_mag)
            setattr(self, 'tcn_gyro_convs%i' % i, tcn_gyro)
            self.tcn_gyro_convs.append(tcn_gyro)
            setattr(self, 'tcn_acc_convs%i' % i, tcn_acc)
            self.tcn_acc_convs.append(tcn_acc)
        
        self.attention = nn.Sequential(
            # nn.Linear(3*feature_channel, 1),
            nn.Linear(feature_channel, 1),
            nn.Tanh()
            )
    
    def forward(self, x):
        
        x_grav = x[:,:,0:3,:]
        x_mag  = x[:,:,3:6,:]
        x_gyro = x[:,:,6:9,:]
        x_acc  = x[:,:,9:12,:]
        
        for i in range(self.scale_num):
            
            out_grav = self.tcn_grav_convs[i](x_grav)
            out_mag  = self.tcn_mag_convs[i](x_mag)
            out_gyro = self.tcn_gyro_convs[i](x_gyro)
            out_acc  = self.tcn_acc_convs[i](x_acc)
            
            attitude           = torch.cat(out_grav, out_mag, dim=2)
            attitude           = attitude.permute(0, 3, 2, 1)
            attention_attitude = self.attention(attitude).squeeze()
            attention_attitude = F.softmax(attention_attitude, dim=2).unsqueeze(-1)
            attitude           = attitude * attention_attitude
            attitude           = attitude.permute(0, 3, 2, 1)
            out_grav           = torch.split(attitude, 2, dim=2)[0]
            out_mag            = torch.split(attitude, 2, dim=2)[1]
            
            if i == 0:
                out_attitude = out_grav + out_mag + out_gyro
                out_dynamic  = out_acc
            else:
                out_attitude = torch.cat([out_attitude, (out_grav + out_mag + out_gyro)], dim=2)
                out_dynamic  = torch.cat([out_dynamic, out_acc], dim=2)
                # out_attitude = out_attitude + (out_grav + out_gyro)
                # out_dynamic  = out_dynamic + out_acc
        
        out = torch.cat([out_attitude, out_dynamic], dim = 2)
        
        return out
    

class Deep_Sensor_Attn_TCN_Transformer(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
                 kernel_size, kernel_size_grav, scale_num, feature_channel_out,
                 multiheads, drop_rate, dataset_name, data_length, num_class):
        
        super(Deep_Sensor_Attn_TCN_Transformer, self).__init__()
        
        self.IMU_fusion_block = IMU_Fusion_Block(input_2Dfeature_channel, input_channel, feature_channel,
                                                 kernel_size_grav, scale_num, dataset_name)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
        #     nn.BatchNorm2d(feature_channel),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(2)
        #     )
        
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
        #     nn.BatchNorm2d(feature_channel),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(2)
        #     )
        
        if input_channel  == 12:
            reduced_channel = 6
        else:
            reduced_channel = 3
        
        self.transition = nn.Sequential(
            nn.Conv1d(feature_channel*(input_channel-reduced_channel)*scale_num, feature_channel_out, 1, 1),
#            nn.Conv1d(feature_channel*(input_channel-reduced_channel), feature_channel_out, 1, 1),
            nn.BatchNorm1d(feature_channel_out),
            nn.ReLU()
            )
        
        self.position_encode = PositionalEncoding(feature_channel_out, drop_rate, data_length)
        
        self.transformer_block1 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        
        self.transformer_block2 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        
        self.global_ave_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.linear = nn.Linear(feature_channel_out, num_class)

    def forward(self, x):
        
        # hidden = None
        batch_size = x.shape[0]
        feature_channel = x.shape[1]
        input_channel = x.shape[2]
        data_length = x.shape[-1]
        
        x, out_attn = self.IMU_fusion_block(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        
        x = x.view(batch_size, -1, data_length)
        x = self.transition(x)
        
        x = self.position_encode(x)
        x = x.permute(0,2,1)
        
        x = self.transformer_block1(x)
        x = self.transformer_block2(x)
        x = x.permute(0,2,1)
        
        x = self.global_ave_pooling(x).squeeze()
        
        output = self.linear(x)
        
        return output, out_attn
    
def train_op(network, EPOCH, BATCH_SIZE, LR,
             train_x, train_y, val_x, val_y, X_test, y_test,
             output_directory_models, log_training_duration, test_split):
    # prepare training_data
    if train_x.shape[0] % BATCH_SIZE == 1:
        drop_last_flag = True
    else:
        drop_last_flag = False
    torch_dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.tensor(train_y).long())
    train_loader = Data.DataLoader(dataset = torch_dataset,
                                    batch_size = BATCH_SIZE,
                                    shuffle = True,
                                    drop_last = drop_last_flag
                                   )
    
    # init lr&train&test loss&acc log
    lr_results = []
    
    loss_train_results = []
    accuracy_train_results = []
    
    loss_validation_results = []
    accuracy_validation_results = []
    macro_f1_val_results        = []
    
    loss_test_results = []
    accuracy_test_results = []
    macro_f1_test_results       = []
    
    # prepare optimizer&scheduler&loss_function
    parameters = ContiguousParams(network.parameters())
    optimizer = torch.optim.Adam(parameters.contiguous(),lr = LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
                                                           patience=5,
                                                           min_lr=LR/10, verbose=True)
    # loss_function = nn.CrossEntropyLoss(reduction='sum')
    loss_function = LabelSmoothingCrossEntropy()
    
    # save init model    
    output_directory_init = output_directory_models+'init_model.pkl'
    torch.save(network.state_dict(), output_directory_init)   # save only the init parameters
    
    # inputs = torch.randn(1, 1, 9, 128).cuda()
    # flops, paras = profile(network, inputs = (inputs))
    # flops, paras = clever_format([flops, paras], "%.3f")
    # print(flops, paras)
    # exit()
    
    training_duration_logs = []
    start_time = time.time()
    for epoch in range (EPOCH):
        
        epoch_tau = epoch+1
        tau = max(1 - (epoch_tau - 1) / 50, 0.5)
        for m in network.modules():
            if hasattr(m, '_update_tau'):
                m._update_tau(tau)
                # print(a)
        
        for step, (x,y) in enumerate(train_loader):
            
            # h_state = None      # for initial hidden state
            
            batch_x = x.cuda()
            batch_y = y.cuda()
            output_bc = network(batch_x)[0]
            
            # cal the sum of pre loss per batch 
            loss = loss_function(output_bc, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # test per epoch
        network.eval()
        test_flag = True
        # loss_train:loss of training set; accuracy_train:pre acc of training set
        loss_train, accuracy_train, _ = get_test_loss_acc(network, loss_function, train_x, train_y, test_split)
        loss_validation, accuracy_validation, macro_f1_val = get_test_loss_acc(network, loss_function, val_x, val_y, test_split)
        loss_test, accuracy_test, macro_f1_test = get_test_loss_acc(network, loss_function, X_test, y_test, test_split)
        test_flag = False
        network.train()
        
        # update lr
        scheduler.step(accuracy_validation)
        lr = optimizer.param_groups[0]['lr']
        
        ######################################dropout#####################################
        # loss_train, accuracy_train = get_loss_acc(network.eval(), loss_function, train_x, train_y, test_split)
        
        # loss_validation, accuracy_validation = get_loss_acc(network.eval(), loss_function, test_x, test_y, test_split)
        
        # network.train()
        ##################################################################################
        
        # log lr&train&validation loss&acc per epoch
        lr_results.append(lr)
        loss_train_results.append(loss_train)    
        accuracy_train_results.append(accuracy_train)
        
        loss_validation_results.append(loss_validation)    
        accuracy_validation_results.append(accuracy_validation)
        macro_f1_val_results.append(macro_f1_val)
        
        loss_test_results.append(loss_test)    
        accuracy_test_results.append(accuracy_test)
        macro_f1_test_results.append(macro_f1_test)
        
        # print training process
        if (epoch+1) % 1 == 0:
            print('Epoch:', (epoch+1), '|lr:', lr,
                  '| train_loss:', loss_train, 
                  '| train_acc:', accuracy_train, 
                  '| validation_loss:', loss_validation, 
                  '| validation_acc:', accuracy_validation)
        
        save_models(network, output_directory_models, 
                    loss_train, loss_train_results, 
                    accuracy_validation, accuracy_validation_results,
                    start_time, training_duration_logs)
    
    # log training time 
    per_training_duration = time.time() - start_time
    log_training_duration.append(per_training_duration)
    
    # save last_model
    output_directory_last = output_directory_models+'last_model.pkl'
    torch.save(network.state_dict(), output_directory_last)   # save only the init parameters
    
    # log history
    history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
                          loss_validation_results, accuracy_validation_results,
                          loss_test_results, accuracy_test_results,
                          output_directory_models)
    
    plot_learning_history(EPOCH, history, output_directory_models)
    
    return(history, per_training_duration, log_training_duration)
