import os
import numpy as np
# hyper-parameters
BATCH_SIZE = 64
EPOCH      = 30
LR         = 0.0005
# mode selection
INFERENCE_DEVICE = 'TEST_CUDA' # DEVICE: 'TEST_CUDA' or 'TEST_CPU'
PATTERN          = 'TRAIN' # 'TRAIN' or 'TEST'
DATASETS    = ['HAPT','Opportunity']
DATASETS    = ['Opportunity']
CLASSIFIERS = ['Deep_Sen_At_TCN_Cr_Br_At_Transformer_torch'] # LR 0.0001, 64
cal_attitude_angle = True
test_split = 50 # when testing, the testing dataset is seperated into 'test_split' pieces

def get_HAPT_dataset_param(CUR_DIR, dataset_name):
    
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = filepath + '\\dataset\\有用UCI HAPT\\数据集\\HAPT_Dataset\\'
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
    DATA_DIR            = filepath + '\\dataset\\有用Opportunity\\'
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
    
    if classifier_name=='Deep_Sensor_Attn_TCN_Transformer_torch': 
        from classifiers import Deep_Sensor_Attn_TCN_Transformer_torch
        # __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
        #          kernel_size, kernel_size_grav, scale_num, feature_channel_out,
        #          multiheads, drop_rate, dataset_name, data_length, num_class)
        return Deep_Sensor_Attn_TCN_Transformer_torch.Deep_Sensor_Attn_TCN_Transformer(1, input_channel, 64, 5, 3, 2, 128, 1, 0.2, dataset_name, data_length, nb_classes), Deep_Sensor_Attn_TCN_Transformer_torch
    
    if classifier_name=='Deep_Sen_At_TCN_Cr_Br_At_Transformer_torch': 
        from classifiers import Deep_Sen_At_TCN_Cr_Br_At_Transformer_torch
        # __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
        #          kernel_size, kernel_size_grav, scale_num, feature_channel_out,
        #          multiheads, drop_rate, dataset_name, data_length, num_class)
        return Deep_Sen_At_TCN_Cr_Br_At_Transformer_torch.Deep_Sen_At_TCN_Cr_Br_At_Transformer(1, input_channel, 64, 5, 3, 2, 128, 1, 0.2, dataset_name, data_length, nb_classes), Deep_Sen_At_TCN_Cr_Br_At_Transformer_torch