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

class SelfAttention(nn.Module):
    """
    # self-attention implementation from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
    Self attention layer for nd
    """
    def __init__(self, n_channels: int, div):
        super(SelfAttention, self).__init__()

        if n_channels > 1:
            self.query = conv1d(n_channels, n_channels//div)
            self.key = conv1d(n_channels, n_channels//div)
        else:
            self.query = conv1d(n_channels, n_channels)
            self.key = conv1d(n_channels, n_channels)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class TemporalAttention(nn.Module):
    """
    Temporal attention module
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x):
        out = self.fc(x).squeeze(2)
        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 0)
        return context

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        activation,
        sa_div,
    ):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(1, filter_num, (1, filter_size), 1, (0,filter_size//2))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (1, filter_size), 1, (0,filter_size//2))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (1, filter_size), 1, (0,filter_size//2))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (1, filter_size), 1, (0,filter_size//2))
        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            filter_num * input_dim,
            hidden_dim,
            enc_num_layers,
            bidirectional=enc_is_bidirectional,
            dropout=dropout_rnn,
            batch_first = True
        )

        self.ta = TemporalAttention(hidden_dim)
        self.sa = SelfAttention(filter_num, sa_div)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # x = x.permute(0,1,3,2)

        # # apply self-attention on each temporal dimension (along sensor and feature dimensions)
        # refined = torch.cat(
        #     [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
        #     dim=-1,
        # )
        # x = refined.permute(3, 0, 1, 2)
        
        x = x.view(x.shape[0], -1, 128)
        x = x.permute(0,2,1)
        
        # x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        outputs, h = self.rnn(x)

        # apply temporal attention on GRU outputs
        # out = self.ta(outputs)
        
        # out = outputs[:,-1,:].squeeze(1)
        out = outputs.view(x.shape[0], 128, -1)[:,-1,:]
        return out


# class Classifier(nn.Module):
#     def __init__(self, hidden_dim, num_class):
#         super(Classifier, self).__init__()
#         self.fc = nn.Linear(hidden_dim, num_class)

#     def forward(self, z):
#         return self.fc(z)


class Attend_And_Discriminate(nn.Module):
    def __init__(
        self,
        # model,
        # dataset,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        train_mode,
        # experiment,
    ):
        super(Attend_And_Discriminate, self).__init__()

        # self.experiment = f"train_{experiment}" if train_mode else experiment
        # self.model = model
        # self.dataset = dataset
        self.hidden_dim = hidden_dim
        # print(paint(f"[STEP 3] Creating {self.model} HAR model ..."))

        self.fe = FeatureExtractor(
            input_dim,
            hidden_dim,
            filter_num,
            filter_size,
            enc_num_layers,
            enc_is_bidirectional,
            dropout,
            dropout_rnn,
            activation,
            sa_div,
        )

        self.dropout = nn.Dropout(dropout_cls)
        # self.classifier = Classifier(hidden_dim, num_class)
        self.linear = nn.Linear(116, 12)
        # self.register_buffer(
        #     "centers", (torch.randn(num_class, self.hidden_dim).cuda())
        # )

    def forward(self, x):
        feature = self.fe(x)
        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        out = self.dropout(feature)
        logits = self.linear(out)
        # logits = self.classifier(out)
        return z, logits

# def init_weights_orthogonal(m):
#     """
#     Orthogonal initialization of layer parameters
#     :param m:
#     :return:
#     """
#     if type(m) == nn.LSTM or type(m) == nn.GRU:
#         for name, param in m.named_parameters():
#             if "weight_ih" in name:
#                 nn.init.orthogonal_(param.data)
#             elif "weight_hh" in name:
#                 nn.init.orthogonal_(param.data)
#             elif "bias" in name:
#                 param.data.fill_(0)

#     elif type(m) == nn.Conv2d or type(m) == nn.Linear:
#         nn.init.orthogonal_(m.weight)
#         m.bias.data.fill_(0)

# class MixUpLoss(nn.Module):
#     """
#     Mixup implementation heavily borrowed from https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py#L42
#     Adapt the loss function `crit` to go with mixup.
#     """

#     def __init__(self, crit, reduction='mean'):
#         super().__init__()
#         if hasattr(crit, 'reduction'):
#             self.crit = crit
#             self.old_red = crit.reduction
#             setattr(self.crit, 'reduction', 'none')
#         self.reduction = reduction

#     def forward(self, output, target):
#         if len(target.size()) == 2:
#             loss1, loss2 = self.crit(output, target[:, 0].long()), self.crit(output, target[:, 1].long())
#             d = loss1 * target[:, 2] + loss2 * (1 - target[:, 2])
#         else:
#             d = self.crit(output, target)
#         if self.reduction == 'mean':
#             return d.mean()
#         elif self.reduction == 'sum':
#             return d.sum()
#         return d

#     def get_old(self):
#         if hasattr(self, 'old_crit'):
#             return self.old_crit
#         elif hasattr(self, 'old_red'):
#             setattr(self.crit, 'reduction', self.old_red)
#             return self.crit

# def mixup_data(x, y, alpha=0.4):

#     """
#     Returns mixed inputs, pairs of targets, and lambda
#     """

#     batch_size = x.shape[0]
#     lam = np.random.beta(alpha, alpha, batch_size)
#     # t = max(t, 1-t)
#     lam = np.concatenate([lam[:, None], 1 - lam[:, None]], 1).max(1)
#     # tensor and cuda version of lam
#     lam = x.new(lam)

#     shuffle = torch.randperm(batch_size).cuda()

#     x1, y1 = x[shuffle], y[shuffle]
#     # out_shape = [bs, 1, 1]
#     out_shape = [lam.size(0)] + [1 for _ in range(len(x1.shape) - 1)]

#     # [bs, temporal, sensor]
#     mixed_x = (x * lam.view(out_shape) + x1 * (1 - lam).view(out_shape))
#     # [bs, 3]
#     y_a_y_b_lam = torch.cat([y[:, None].float(), y1[:, None].float(), lam[:, None].float()], 1)

#     return mixed_x, y_a_y_b_lam

# def compute_center_loss(features, centers, targets):

#     # centerloss implementation from https://github.com/louis-she/center-loss.pytorch/blob/master/loss.py

#     features = features.view(features.size(0), -1)
#     target_centers = centers[targets]
#     criterion = torch.nn.MSELoss()
#     center_loss = criterion(features, target_centers)
#     return center_loss


# def get_center_delta(features, centers, targets, alpha):
#     # implementation equation (4) in the center-loss paper
#     features = features.view(features.size(0), -1)
#     targets, indices = torch.sort(targets)
#     target_centers = centers[targets]
#     features = features[indices]

#     delta_centers = target_centers - features
#     uni_targets, indices = torch.unique(
#             targets.cpu(), sorted=True, return_inverse=True)

#     uni_targets = uni_targets.cuda()
#     indices = indices.cuda()

#     delta_centers = torch.zeros(
#         uni_targets.size(0), delta_centers.size(1)
#     ).cuda().index_add_(0, indices, delta_centers)

#     targets_repeat_num = uni_targets.size()[0]
#     uni_targets_repeat_num = targets.size()[0]
#     targets_repeat = targets.repeat(
#             targets_repeat_num).view(targets_repeat_num, -1)
#     uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
#             1, uni_targets_repeat_num)
#     same_class_feature_count = torch.sum(
#             targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

#     delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
#     result = torch.zeros_like(centers)
#     result[uni_targets, :] = delta_centers
#     return result

# def train_op(network, EPOCH, BATCH_SIZE, LR,
#              train_x, train_y, val_x, val_y, X_test, y_test,
#              output_directory_models, log_training_duration, test_split):
#     # prepare training_data
#     if train_x.shape[0] % BATCH_SIZE == 1:
#         drop_last_flag = True
#     else:
#         drop_last_flag = False
#     torch_dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.tensor(train_y).long())
#     train_loader = Data.DataLoader(dataset = torch_dataset,
#                                     batch_size = BATCH_SIZE,
#                                     shuffle = True,
#                                     drop_last = drop_last_flag
#                                    )
    
#     # init lr&train&test loss&acc log
#     lr_results = []
    
#     loss_train_results = []
#     accuracy_train_results = []
    
#     loss_validation_results = []
#     accuracy_validation_results = []
#     macro_f1_val_results        = []
    
#     loss_test_results = []
#     accuracy_test_results = []
#     macro_f1_test_results       = []
    
#     # prepare optimizer&scheduler&loss_function
#     # network.apply(init_weights_orthogonal)
#     parameters = network.parameters()
#     optimizer = torch.optim.Adam(parameters,lr = LR)
#     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
#     #                                                        patience=5,
#     #                                                        min_lr=0.00001, verbose=True)
#     scheduler = torch.optim.lr_scheduler.StepLR(
#                    optimizer, step_size=10, gamma=0.9)
#     criterion = nn.CrossEntropyLoss(reduction='sum')
#     # loss_function = LabelSmoothingCrossEntropy()
    
#     # save init model    
#     output_directory_init = output_directory_models+'init_model.pkl'
#     torch.save(network.state_dict(), output_directory_init)   # save only the init parameters
    
#     training_duration_logs = []
#     start_time = time.time()
    
#     mixup = False
    
    
#     for epoch in range (EPOCH):
        
#         for step, (x,y) in enumerate(train_loader):
            
#             # h_state = None      # for initial hidden state
            
#             batch_x = x.cuda()
#             batch_y = y.cuda()
            
#             # centers = network.centers
            
#             # if mixup == True:
#             #     batch_x, batch_y_mixup = mixup_data(batch_x, batch_y, 0.8)
            
#             z, logits              = network(batch_x)
            
#             # cal the sum of pre loss per batch
#             # if mixup == True:
#             #     loss_function    = MixUpLoss(criterion)
#             #     loss             = loss_function(logits, batch_y_mixup)
#             # else:
#             #     loss             = criterion(logits, batch_y)
            
#             loss             = criterion(logits, batch_y) # 后删
            
#             # center_loss      = compute_center_loss(z, centers, batch_y)
#             # loss             = loss + 0 * center_loss
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             # center_deltas     = get_center_delta(z.data, centers, batch_y, 0.001)
#             # network.centers   = centers - center_deltas
    
#             # if mixup == True:
#             #     loss_function = loss_function.get_old()
        
#         # test per epoch
#         network.eval()
#         # loss_train:loss of training set; accuracy_train:pre acc of training set
#         loss_train, accuracy_train, _ = get_test_loss_acc(network, criterion, train_x, train_y, test_split)        
#         loss_validation, accuracy_validation, macro_f1_val = get_test_loss_acc(network, criterion, val_x, val_y, test_split) 
#         loss_test, accuracy_test, macro_f1_test = get_test_loss_acc(network, criterion, X_test, y_test, test_split)
#         network.train()  
        
#         # update lr
#         scheduler.step()
#         lr = optimizer.param_groups[0]['lr']
        
#         ######################################dropout#####################################
#         # loss_train, accuracy_train = get_loss_acc(network.eval(), loss_function, train_x, train_y, test_split)
        
#         # loss_validation, accuracy_validation = get_loss_acc(network.eval(), loss_function, test_x, test_y, test_split)
        
#         # network.train()
#         ##################################################################################
        
#         # log lr&train&validation loss&acc per epoch
#         lr_results.append(lr)
#         loss_train_results.append(loss_train)    
#         accuracy_train_results.append(accuracy_train)
        
#         loss_validation_results.append(loss_validation)    
#         accuracy_validation_results.append(accuracy_validation)
#         macro_f1_val_results.append(macro_f1_val)
        
#         loss_test_results.append(loss_test)    
#         accuracy_test_results.append(accuracy_test)
#         macro_f1_test_results.append(macro_f1_test)
        
#         # print training process
#         if (epoch+1) % 1 == 0:
#             print('Epoch:', (epoch+1), '|lr:', lr,
#                   '| train_loss:', loss_train, 
#                   '| train_acc:', accuracy_train, 
#                   '| validation_loss:', loss_validation, 
#                   '| validation_acc:', accuracy_validation)
        
#         save_models(network, output_directory_models, 
#                     loss_train, loss_train_results, 
#                     accuracy_validation, accuracy_validation_results,
#                     start_time, training_duration_logs)
    
#     # log training time 
#     per_training_duration = time.time() - start_time
#     log_training_duration.append(per_training_duration)
    
#     # save last_model
#     output_directory_last = output_directory_models+'last_model.pkl'
#     torch.save(network.state_dict(), output_directory_last)   # save only the init parameters
    
#     # log history
#     history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
#                           loss_validation_results, accuracy_validation_results,
#                           loss_test_results, accuracy_test_results,
#                           output_directory_models)
    
#     plot_learning_history(EPOCH, history, output_directory_models)
    
#     return(history, per_training_duration, log_training_duration)


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
    parameters = network.parameters()
    optimizer = torch.optim.Adam(parameters,lr = LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
                                                           patience=5,
                                                           min_lr=0.00002, verbose=True)
    # loss_function = nn.CrossEntropyLoss(reduction='sum')
    loss_function = LabelSmoothingCrossEntropy()
    
    # save init model    
    output_directory_init = output_directory_models+'init_model.pkl'
    torch.save(network.state_dict(), output_directory_init)   # save only the init parameters
    
    training_duration_logs = []
    start_time = time.time()
    for epoch in range (EPOCH):
        
        for step, (x,y) in enumerate(train_loader):
            
            # h_state = None      # for initial hidden state
            
            batch_x = x.cuda()
            batch_y = y.cuda()
            z, logits = network(batch_x)
            
            # cal the sum of pre loss per batch 
            loss = loss_function(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # test per epoch
        network.eval()
        # loss_train:loss of training set; accuracy_train:pre acc of training set
        loss_train, accuracy_train, _ = get_test_loss_acc(network, loss_function, train_x, train_y, test_split)        
        loss_validation, accuracy_validation, macro_f1_val = get_test_loss_acc(network, loss_function, val_x, val_y, test_split) 
        loss_test, accuracy_test, macro_f1_test = get_test_loss_acc(network, loss_function, X_test, y_test, test_split)
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