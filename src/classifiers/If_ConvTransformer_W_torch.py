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
        # map k-dimentional input to k*heads dimentions
        self.tokeys    = nn.Linear(k, k * heads, bias = False)
        self.toqueries = nn.Linear(k, k * heads, bias = False)
        self.tovalues  = nn.Linear(k, k * heads, bias = False)
        # set dropout
        self.dropout_attention = nn.Dropout(drop_rate)
        # squeeze dimention to k
        self.unifyheads = nn.Linear(heads * k, k)
        
    def forward(self, x):
        
        b, t, k = x.size()
        h = self.heads
        queries = self.toqueries(x).view(b, t, h, k)
        keys    = self.tokeys(x).view(b, t, h, k)
        values  = self.tovalues(x).view(b, t, h, k)
        # squeeze head into batch dimension
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys    = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values  = values.transpose(1, 2).contiguous().view(b * h, t, k)
        # normalize the dot products
        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))
        # matrix multiplication
        dot  = torch.bmm(queries, keys.transpose(1,2))
        # softmax normalization
        dot = F.softmax(dot, dim=2)
        dot = self.dropout_attention(dot)
        out = torch.bmm(dot, values).view(b, h, t, k)
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h*k)
        
        return self.unifyheads(out) # (b, t, k)

def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    """
    Create and initialize a `nn.Conv1d` layer with spectral normalization.
    """
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    # return spectral_norm(conv)
    return conv

class SelfAttention_Branch(nn.Module):
    
    def __init__(self, n_channels: int, drop_rate = 0, div = 1):
        super(SelfAttention_Branch, self).__init__()

        self.n_channels = n_channels

        if n_channels > 1:
            self.query = conv1d(n_channels, n_channels//div)
            self.key = conv1d(n_channels, n_channels//div)
        else:
            self.query = conv1d(n_channels, n_channels)
            self.key = conv1d(n_channels, n_channels)
        self.value = conv1d(n_channels, n_channels)
        self.dropout_attention = nn.Dropout(drop_rate)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        
        x = x.permute(0,2,1)
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = (self.query(x) / (self.n_channels ** (1/4))), (self.key(x) / (self.n_channels ** (1/4))), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        beta = self.dropout_attention(beta)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous().permute(0,2,1)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads, drop_rate):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(k, heads = heads, drop_rate = drop_rate)
        self.norm1 = nn.BatchNorm1d(k)

        self.mlp = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )
        self.norm2 = nn.BatchNorm1d(k)
        self.dropout_forward = nn.Dropout(drop_rate)

    def forward(self, x):
        
        # perform self-attention
        attended = self.attention(x)
        attended = attended + x
        attended = attended.permute(0,2,1)
        x = self.norm1(attended).permute(0,2,1)
        # feedforward
        feedforward = self.mlp(x)
        
        feedforward = feedforward + x
        feedforward = feedforward.permute(0,2,1)
        
        return self.dropout_forward(self.norm2(feedforward).permute(0,2,1))

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
            
            padding_grav      = (kernel_size_grav - 1) * dilation_num_grav
            kernel_size_gyro  = padding_grav
            kernel_size_acc   = padding_grav + 1
            
            tcn_grav = nn.Sequential(
                weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                      (1,kernel_size_grav), 1, (0,padding_grav), 
                                      dilation=dilation_num_grav)),
                Chomp2d(padding_grav),
                nn.ReLU(),
                )
            
            
            if kernel_size_gyro == 1:
                tcn_gyro = nn.Sequential(
                    weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                          (1,1), 1, (0,0), 
                                          dilation=1)),
                    nn.ReLU(),
                    )
            else:
                tcn_gyro = nn.Sequential(
                    weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                          (1,kernel_size_gyro), 1, (0,(kernel_size_gyro-1)*1), 
                                          dilation=1)),
                    Chomp2d((kernel_size_gyro-1)*1),
                    nn.ReLU(),
                    )
            
            tcn_acc = nn.Sequential(
                weight_norm(nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                      (1,kernel_size_acc), 1, (0,(kernel_size_acc-1)*1), 
                                      dilation=1)),
                Chomp2d((kernel_size_acc-1)*1),
                nn.ReLU(),
                )
            
            setattr(self, 'tcn_grav_convs%i' % i, tcn_grav)
            self.tcn_grav_convs.append(tcn_grav)
            setattr(self, 'tcn_gyro_convs%i' % i, tcn_gyro)
            self.tcn_gyro_convs.append(tcn_gyro)
            setattr(self, 'tcn_acc_convs%i' % i, tcn_acc)
            self.tcn_acc_convs.append(tcn_acc)
        
        self.attention = nn.Sequential(
                nn.Linear(3*feature_channel, 1),
                # nn.Tanh()
                nn.PReLU()
                )
        
    def forward(self, x):
        
        x_grav = x[:,:,0:3,:]
        x_gyro = x[:,:,3:6,:]
        x_acc  = x[:,:,6:9,:]
    
        for i in range(self.scale_num):
            
            out_grav = self.tcn_grav_convs[i](x_grav).unsqueeze(4)
            out_gyro = self.tcn_gyro_convs[i](x_gyro).unsqueeze(4)
            out_acc  = self.tcn_acc_convs[i](x_acc)
            
            if i == 0:
                out_attitude = torch.cat([out_grav, out_gyro], dim=4)
                out_dynamic  = out_acc
            else:
                out_attitude = torch.cat([out_attitude, out_grav], dim=4)
                out_attitude = torch.cat([out_attitude, out_gyro], dim=4)
                out_dynamic  = torch.cat([out_dynamic, out_acc], dim=2)
                
        # (batch_size, time_length, sensor_num*scale_num, 3(xyz), feature_chnnl)
        out_attitude = out_attitude.permute(0,3,4,2,1)
        # (batch_size, time_length, sensor_num*scale_num, 3(xyz)*feature_chnnl)
        out_attitude = out_attitude.reshape(out_attitude.shape[0], out_attitude.shape[1], out_attitude.shape[2], -1)
        # time-step-wise sensor attention, sensor_attn:(batch_size, time_length, sensor_num*scale_num, 1)
        sensor_attn  = self.attention(out_attitude).squeeze(3)
        sensor_attn  = F.softmax(sensor_attn, dim=2).unsqueeze(-1)
        out_attitude = sensor_attn * out_attitude
        
        # used for normalization
        norm_num     = torch.mean(sensor_attn.squeeze(-1), dim=1)
        norm_num     = torch.pow(norm_num, 2)
        norm_num     = torch.sqrt(torch.sum(norm_num, dim=1))
        norm_num     = (pow(self.scale_num,0.5)/norm_num).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        out_attitude = out_attitude * norm_num
        
        # (batch_size, time_length, sensor_num*scale_num, 3(xyz), feature_chnnl)
        out_attitude = out_attitude.reshape(out_attitude.shape[0], out_attitude.shape[1], out_attitude.shape[2], 3, -1)
        # (batch_size, time_length, sensor_num*scale_num*3(xyz), feature_chnnl)
        out_attitude = out_attitude.reshape(out_attitude.shape[0], out_attitude.shape[1], out_attitude.shape[2]*3, -1)
        # (batch_size, feature_chnnl, sensor_num*scale_num*3(xyz), time_length)
        out_attitude = out_attitude.permute(0,3,2,1)
        
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

class If_ConvTransformer_W(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
                 kernel_size, kernel_size_grav, scale_num, feature_channel_out,
                 multiheads, drop_rate, dataset_name, data_length, num_class):
        
        super(If_ConvTransformer_W, self).__init__()
       
        self.feature_channel  = feature_channel
        self.scale_num        = scale_num
        
        self.IMU_fusion_blocks     = []
        for i in range(input_channel//9):
            IMU_fusion_block   = IMU_Fusion_Block(input_2Dfeature_channel, input_channel, feature_channel,
                                                  kernel_size_grav, scale_num, dataset_name)
            setattr(self, 'IMU_fusion_blocks%i' % i, IMU_fusion_block)
            self.IMU_fusion_blocks.append(IMU_fusion_block)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        if input_channel  == 12:
            reduced_channel = 6
        else:
            reduced_channel = 3
        
        self.norm_conv4  = nn.LayerNorm(feature_channel)

        self.sa         = SelfAttention_Branch(feature_channel, drop_rate = drop_rate)

        self.transition = nn.Sequential(
            nn.Conv1d(feature_channel*(9-reduced_channel)*scale_num*input_channel//9, feature_channel_out, 1, 1),
            nn.BatchNorm1d(feature_channel_out),
            nn.ReLU()
            )

        self.position_encode = PositionalEncoding(feature_channel_out, drop_rate, data_length)
        
        self.transformer_block1 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        
        self.transformer_block2 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        
        self.global_ave_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.linear = nn.Linear(feature_channel_out, num_class)
        
        self.register_buffer(
            "centers", (torch.randn(num_class, feature_channel_out).cuda())
        )

    def forward(self, x):
        
        # hidden = None
        batch_size      = x.shape[0]
        input_channel   = x.shape[2]
        data_length     = x.shape[-1]
        IMU_num         = input_channel//9
        x_input         = x
        
        for i in range(IMU_num):
            x_cur_IMU, cur_sensor_attn   = self.IMU_fusion_blocks[i](x_input[:,:,i*9:(i+1)*9,:])
            if i == 0:
                x        = x_cur_IMU
                out_attn = cur_sensor_attn
            else:
                x        = torch.cat((x, x_cur_IMU), 2)
                out_attn = torch.cat((out_attn, cur_sensor_attn), 2)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # [batch_size, fea_dims, sensor_chnnl, data_len]
        
        x = x.permute(0, 3, 2, 1)
        x = self.norm_conv4(x).permute(0, 3, 2, 1)
        
        x = x.permute(0,3,2,1).reshape(batch_size*data_length, -1, self.feature_channel)
        x = self.sa(x).reshape(batch_size, data_length, -1, self.feature_channel)
        x = x.permute(0,3,2,1).reshape(batch_size, -1, data_length)
        
        x = self.transition(x)
        
        x = self.position_encode(x)
        x = x.permute(0,2,1)
        
        x = self.transformer_block1(x)
        x = self.transformer_block2(x)
        x = x.permute(0,2,1)
        
        x = self.global_ave_pooling(x).squeeze(-1)
        
        z = x.div(
            torch.norm(x, p=2, dim=1, keepdim=True).expand_as(x)
        )
        
        output = self.linear(x)
        
        return output, z
    
def init_weights_orthogonal(m):
    """
    Orthogonal initialization of layer parameters
    :param m:
    :return:
    """
    if type(m) == nn.LSTM or type(m) == nn.GRU:
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    elif type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)
        # m.bias.data.fill_(0)

class MixUpLoss(nn.Module):

    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        self.reduction = reduction

    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = self.crit(output, target[:, 0].long()), self.crit(output, target[:, 1].long())
            d = loss1 * target[:, 2] + loss2 * (1 - target[:, 2])
        else:
            d = self.crit(output, target)
        if self.reduction == 'mean':
            return d.mean()
        elif self.reduction == 'sum':
            return d.sum()
        return d

    def get_old(self):
        if hasattr(self, 'old_crit'):
            return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit

def mixup_data(x, y, alpha=0.4):

    """
    Returns mixed inputs, pairs of targets, and lambda
    """

    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha, batch_size)
    # t = max(t, 1-t)
    lam = np.concatenate([lam[:, None], 1 - lam[:, None]], 1).max(1)
    # tensor and cuda version of lam
    lam = x.new(lam)

    shuffle = torch.randperm(batch_size).cuda()

    x1, y1 = x[shuffle], y[shuffle]
    # out_shape = [bs, 1, 1]
    out_shape = [lam.size(0)] + [1 for _ in range(len(x1.shape) - 1)]

    # [bs, temporal, sensor]
    mixed_x = (x * lam.view(out_shape) + x1 * (1 - lam).view(out_shape))
    # [bs, 3]
    y_a_y_b_lam = torch.cat([y[:, None].float(), y1[:, None].float(), lam[:, None].float()], 1)

    return mixed_x, y_a_y_b_lam

def compute_center_loss(features, centers, targets):

    features = features.view(features.size(0), -1)
    target_centers = centers[targets]
    criterion = torch.nn.MSELoss()
    center_loss = criterion(features, target_centers)
    return center_loss


def get_center_delta(features, centers, targets, alpha):
    # implementation equation (4) in the center-loss paper
    features = features.view(features.size(0), -1)
    targets, indices = torch.sort(targets)
    target_centers = centers[targets]
    features = features[indices]

    delta_centers = target_centers - features
    uni_targets, indices = torch.unique(
            targets.cpu(), sorted=True, return_inverse=True)

    uni_targets = uni_targets.cuda()
    indices = indices.cuda()

    delta_centers = torch.zeros(
        uni_targets.size(0), delta_centers.size(1)
    ).cuda().index_add_(0, indices, delta_centers)

    targets_repeat_num = uni_targets.size()[0]
    uni_targets_repeat_num = targets.size()[0]
    targets_repeat = targets.repeat(
            targets_repeat_num).view(targets_repeat_num, -1)
    uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
            1, uni_targets_repeat_num)
    same_class_feature_count = torch.sum(
            targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

    delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
    result = torch.zeros_like(centers)
    result[uni_targets, :] = delta_centers
    return result

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
    network.apply(init_weights_orthogonal)
    parameters = network.parameters()
    optimizer = torch.optim.Adam(parameters,lr = LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=10, gamma=0.9)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # loss_function = LabelSmoothingCrossEntropy()
    
    # save init model    
    output_directory_init = os.path.join(output_directory_models, 'init_model.pkl')
    torch.save(network.state_dict(), output_directory_init)   # save only the init parameters
    
    training_duration_logs = []
    start_time = time.time()
    
    # super param
    mixup = True
    alpha = 0.8
    beta  = 0.0003
    lr_cent = 0.001
    #############
    
    for epoch in range (EPOCH):
        
        for step, (x,y) in enumerate(train_loader):
            
            # h_state = None      # for initial hidden state
            
            batch_x = x.cuda()
            batch_y = y.cuda()
            
            centers = network.centers
            
            if mixup == True:
                batch_x, batch_y_mixup = mixup_data(batch_x, batch_y, alpha)
            
            logits, z            = network(batch_x)
            
            # cal the sum of pre loss per batch
            if mixup == True:
                loss_function    = MixUpLoss(criterion)
                loss             = loss_function(logits, batch_y_mixup)
            else:
                loss             = criterion(logits, batch_y)
            
            center_loss      = compute_center_loss(z, centers, batch_y)
            loss             = loss + beta * center_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            center_deltas     = get_center_delta(z.data, centers, batch_y, lr_cent)
            network.centers   = centers - center_deltas
    
            if mixup == True:
                loss_function = loss_function.get_old()
        
        # test per epoch
        network.eval()
        # loss_train:loss of training set; accuracy_train:pre acc of training set
        loss_train, accuracy_train, _ = get_test_loss_acc(network, criterion, train_x, train_y, test_split)
        loss_validation, accuracy_validation, macro_f1_val = get_test_loss_acc(network, criterion, val_x, val_y, test_split)
        loss_test, accuracy_test, macro_f1_test = get_test_loss_acc(network, criterion, X_test, y_test, test_split)
        network.train()  
        
        # update lr
        scheduler.step()
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
    output_directory_last = os.path.join(output_directory_models, 'last_model.pkl')
    torch.save(network.state_dict(), output_directory_last)   # save only the init parameters
    
    # log history
    history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
                          loss_validation_results, accuracy_validation_results,
                          loss_test_results, accuracy_test_results,
                          output_directory_models)
    
    plot_learning_history(EPOCH, history, output_directory_models)
    
    return(history, per_training_duration, log_training_duration)