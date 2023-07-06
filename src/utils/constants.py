import os
import numpy as np
import argparse

def parse_args():
    # The training options
      parser = argparse.ArgumentParser(description='If_ConvTransformer')
      
      parser.add_argument('--PATTERN', type=str, default='TRAIN',
                          help='pattern: TRAIN, TEST')
      parser.add_argument('--DATASETS', nargs='+', default=['Opportunity'],
                          help='dataset name: HAPT, Opportunity')
      parser.add_argument('--CLASSIFIERS', nargs='+', default=['If_ConvTransformer_W_torch'],
                          help='classifier name: If_ConvTransformer_torch, If_ConvTransformer_W_torch')
      parser.add_argument('--BATCH_SIZE', type=int, default=64,
                          help='training batch size: Oppo: 64 HAPT: 128')
      parser.add_argument('--EPOCH', type=int, default=100,
                          help='training epoches: HAPT: 30, Oppo: 100')
      parser.add_argument('--LR', type=float, default=0.0005,
                          help='learning rate: HAPT: 0.0002, Oppo: 0.0005')
      parser.add_argument('--cal_attitude_angle', type=bool, default=True,
                          help='correct the rotation angle')
      parser.add_argument('--test_split', type=int, default=50,
                          help='the testing dataset is seperated into test_split pieces in the inference process')
      parser.add_argument('--INFERENCE_DEVICE', type=str, default='TEST_CUDA',
                          help='inference device: TEST_CUDA, TEST_CPU')
      
      args = parser.parse_args()
      return args

def get_HAPT_dataset_param(CUR_DIR, dataset_name):
    
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = filepath + '\\dataset\\UCI HAPT\\HAPT_Dataset\\'
    MODELS_COMP_LOG_DIR = CUR_DIR + '\\logs\\'+ dataset_name +'\\classifiers_comparison\\'
    ACT_LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", 
                  "SITTING", "STANDING", "LAYING", 
                  "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE",
                  "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]
    ActID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    TRAIN_SUBJECTS_ID = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
    TEST_SUBJECTS_ID = [2, 4, 9, 10, 12, 13, 18, 20, 24]
    ALL_SUBJECTS_ID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
                       14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
                       25, 26, 27, 28, 29, 30]
    WINDOW_SIZE = 128 # default: 128
    OVERLAP = 64 # default: 64
    INPUT_CHANNEL = 9
    POS_NUM = 1
    separate_gravity_flag = True
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID, TRAIN_SUBJECTS_ID,\
           TEST_SUBJECTS_ID, WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, POS_NUM, separate_gravity_flag

def get_Opportunity_dataset_param(CUR_DIR, dataset_name):
    
    (filepath, _)       = os.path.split(CUR_DIR)
    DATA_DIR            = filepath + '\\dataset\\Opportunity\\'
    MODELS_COMP_LOG_DIR = CUR_DIR + '\\logs\\'+ dataset_name +'\\classifiers_comparison\\'
    SUBJECTS                = [1,2,3,4]
    TRIALS                  = [1,2,3,4,5]
    SELEC_LABEL             = 'MID_LABEL_COL' # 'LOCO_LABEL_COL', 'MID_LABEL_COL', 'HI_LABEL_COL'
    ACT_LABELS              = ['null', 'Open_Door_1', 'Open_Door_2', 'Close_Door_1', 'Close_Door_2', 'Open_Fridge',
                               'Close_Fridge', 'Open_Dishwasher', 'Close_Dishwasher', 'Open Drawer1','Close Drawer1',
                               'Open_Drawer2','Close_Drawer2', 'Open_Drawer3', 'Close_Drawer3', 'Clean_Table',
                               'Drink_Cup', 'Toggle_Switch']
    ACT_ID                  = (np.arange(18)).tolist()
    TRAIN_SUBJECTS_ID       = [1]
    TRAIN_SUBJECTS_TRIAL_ID = [1,2,3,4,5]
    WINDOW_SIZE             = 24
    OVERLAP                 = 12
    INPUT_CHANNEL           = 63 # 63, 42
    POS_NUM                 = 7
    separate_gravity_flag   = True
    to_NED_flag             = True
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS, TRIALS, SELEC_LABEL,\
           ACT_LABELS, ACT_ID, TRAIN_SUBJECTS_ID, TRAIN_SUBJECTS_TRIAL_ID,\
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, POS_NUM, separate_gravity_flag, to_NED_flag

def create_classifier(dataset_name, classifier_name, input_channel, POS_NUM,
                      data_length, nb_classes):
    
    if classifier_name=='If_ConvTransformer_torch': 
        from classifiers import If_ConvTransformer_torch
        # __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
        #          kernel_size, kernel_size_grav, scale_num, feature_channel_out,
        #          multiheads, drop_rate, dataset_name, data_length, num_class)
        return If_ConvTransformer_torch.If_ConvTransformer(1, input_channel, 64, 5, 3, 2, 128, 1, 0.2, dataset_name, data_length, nb_classes), If_ConvTransformer_torch
    
    if classifier_name=='If_ConvTransformer_W_torch': 
        from classifiers import If_ConvTransformer_W_torch
        # __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
        #          kernel_size, kernel_size_grav, scale_num, feature_channel_out,
        #          multiheads, drop_rate, dataset_name, data_length, num_class)
        return If_ConvTransformer_W_torch.If_ConvTransformer_W(1, input_channel, 64, 5, 3, 2, 128, 1, 0.2, dataset_name, data_length, nb_classes), If_ConvTransformer_W_torch