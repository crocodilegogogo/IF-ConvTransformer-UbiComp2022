import os
import sys
import numpy as np 
from copy import deepcopy
from time import gmtime, strftime

from scipy.interpolate import interp1d
from scipy.fftpack import fft
from utils.load_Opportunity_dataset.preprocessing import *
# from preprocessing import *
import pandas as pd
from typing import Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# separate_gravity_flag = True
# cal_attitude_angle    = True
# scaler                = "normalize"
# window_size           = 400
# overlap               = 200
# TRAIN_SUBJECTS        = [1, 2, 3, 4, 6, 7, 8]

# read_data_dir = 'Per_subject_npy'

# dataList = os.listdir(read_data_dir)

# gtType = ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]
# idxList = range(len(gtType))
# gtIdxDict = dict(zip(gtType, idxList))
# idxGtDict = dict(zip(idxList, gtType))

# ACT_LABELS    = ["a","b","c","d","e","f","g","h","i"]
# id_sub_List = range(len(subjects))
# subIdxDict  = dict(zip(subjects, id_sub_List))

def preprocess_signal(signal: pd.DataFrame, window_size, overlap) -> pd.DataFrame:
    _signal = signal.copy()
    of = Preprocess()
    # _signal = of.apply_filter(_signal, filter="median")
    # _signal = of.apply_filter(_signal, filter="butterworth")
    _signal = of.segment_signal(_signal, window_size, overlap)
    return _signal


def scale(
    signal: pd.DataFrame, scaler="normalize", minmax_range: Optional[Tuple[int, int]] = (0, 1)
) -> pd.DataFrame:
    if scaler == "normalize":
        signal = StandardScaler().fit_transform(signal) # 这个是对xyz三轴同时做归一化
        return pd.DataFrame(signal, columns=["x", "y", "z"])
    elif scaler == "minmax":
        signal = MinMaxScaler(feature_range=minmax_range).fit_transform(signal)
        return pd.DataFrame(signal, columns=["x", "y", "z"])

def preprocess_raw_data(DATA_DIR, SUBJECTS, TRIALS, SELEC_LABEL,
                        TRAIN_SUBJECTS_ID, TRAIN_SUBJECTS_TRIAL_ID,
                        window_size, overlap, cal_attitude_angle,
                        scaler, separate_gravity_flag=True,
                        to_NED_flag = True):
# def preprocess_raw_data(read_data_dir, SUBJECTS, TRAIN_SUBJECTS_ID, window_size, overlap, cal_attitude_angle,
#                         scaler, separate_gravity_flag=True, to_NED_flag=True):

    X_train = np.array([])
    Y_train = np.array([])
    X_test  = np.array([])
    Y_test  = np.array([])
    
    # dataList = os.listdir(read_data_dir)
    df             = pd.read_csv(os.path.join(DATA_DIR, 'clean_opp.csv'))
    FEATURES       = [str(i) for i in range(97)]
    LOCO_LABEL_COL = 97
    MID_LABEL_COL  = 98
    HI_LABEL_COL   = 99
    SUBJECT_ID     = 100
    RUN_ID         = 101
    
    if SELEC_LABEL   == 'MID_LABEL_COL':
        SELEC_LABEL  = MID_LABEL_COL
    elif SELEC_LABEL == 'LOCO_LABEL_COL':
        SELEC_LABEL  = LOCO_LABEL_COL
    elif SELEC_LABEL == 'HI_LABEL_COL':
        SELEC_LABEL  = HI_LABEL_COL
    
    # df            = df[df[str(HI_LABEL_COL)] != 0]
    LABELS         = np.unique(df.values[:,SELEC_LABEL]).tolist()
    
    for sub_id in SUBJECTS:
        
        cur_sub_frag = df[df[str(SUBJECT_ID)] == sub_id]
        
        # normalization per trial
        for trail_id in TRIALS:
            
            cur_trail = cur_sub_frag[cur_sub_frag[str(RUN_ID)] == trail_id].values
            
            # preprocess the IMU data without shoes
            for pos_id in range(5):
                cur_pos_frag       = cur_trail[:, pos_id*13 : (pos_id+1)*13]
                
                cur_pos_acc        = pd.DataFrame(cur_pos_frag[:,0:3], columns = ["x", "y", "z"])
                cur_pos_gyro       = pd.DataFrame(cur_pos_frag[:,3:6], columns = ["x", "y", "z"])
                cur_pos_quaternion = cur_pos_frag[:, 9:13]/1000
                cur_pos_quaternion = NED_R(cur_pos_quaternion)

                if separate_gravity_flag == True:
                    # separate gravity and linear acc
                    pre_obj = Preprocess(30)
                    cur_pos_acc_body, cur_pos_grav = pre_obj.separate_gravity(cur_pos_acc, cal_attitude_angle) # seperate gravity from acc
                    # Correct the IMU data to NED
                    if to_NED_flag == True:
                        cur_pos_grav       = pre_threeaxis_data(to_NED_flag, cur_pos_grav, cur_pos_quaternion)
                        cur_pos_acc_body   = pre_threeaxis_data(to_NED_flag, cur_pos_acc_body, cur_pos_quaternion)
                        cur_pos_gyro       = pre_threeaxis_data(to_NED_flag, cur_pos_gyro, cur_pos_quaternion)
                    # norm per sensor
                    cur_pos_grav     = scale(cur_pos_grav)
                    cur_pos_acc_body = scale(cur_pos_acc_body)
                    cur_pos_gyro     = scale(cur_pos_gyro)
                
                    cur_pos_data       = np.concatenate((cur_pos_grav, cur_pos_gyro, cur_pos_acc_body), axis=1)
                    
                elif separate_gravity_flag == False:
                    if to_NED_flag == True:
                        cur_pos_acc  = pre_threeaxis_data(to_NED_flag, cur_pos_acc, cur_pos_quaternion)
                        cur_pos_gyro = pre_threeaxis_data(to_NED_flag, cur_pos_gyro, cur_pos_quaternion)
                    # norm per sensor
                    cur_pos_acc     = scale(cur_pos_acc)
                    cur_pos_gyro    = scale(cur_pos_gyro)
                    
                    cur_pos_data    = np.concatenate((cur_pos_acc, cur_pos_gyro), axis=1)
                
                if pos_id == 0:
                    cur_trail_frag = cur_pos_data
                else:
                    cur_trail_frag = np.concatenate((cur_trail_frag, cur_pos_data), axis=1)
            # preprocess the IMU data of shoes
            for shoe_id in range(2):
                cur_shoe_frag       = cur_trail[:, (5*13+shoe_id*16) : (5*13+(shoe_id+1)*16)]
                
                cur_shoe_acc        = pd.DataFrame(cur_shoe_frag[:,6:9], columns = ["x", "y", "z"])
                cur_shoe_gyro       = pd.DataFrame(cur_shoe_frag[:,9:12], columns = ["x", "y", "z"])
                cur_shoe_quaternion = cur_shoe_frag[:,0:3]*2*math.pi/360
                
                if separate_gravity_flag == True:
                    # separate gravity and linear acc
                    pre_obj = Preprocess(30)
                    cur_shoe_acc_body, cur_shoe_grav = pre_obj.separate_gravity(cur_shoe_acc, cal_attitude_angle) # seperate gravity from acc
                    if to_NED_flag == True:
                        cur_shoe_grav, cur_shoe_acc_body, cur_shoe_gyro = correct_orientation9(cur_shoe_grav, \
                                                                                               cur_shoe_acc_body, \
                                                                                               cur_shoe_gyro, cur_shoe_quaternion)
                    # norm per sensor
                    cur_shoe_grav     = scale(cur_shoe_grav)
                    cur_shoe_acc_body = scale(cur_shoe_acc_body)
                    cur_shoe_gyro     = scale(cur_shoe_gyro)
                    
                    cur_shoe_data     = np.concatenate((cur_shoe_grav, cur_shoe_gyro, cur_shoe_acc_body), axis=1)
                
                if separate_gravity_flag == False:
                    if to_NED_flag == True:
                        cur_shoe_acc, cur_shoe_gyro = correct_orientation6(cur_shoe_acc, cur_shoe_gyro, cur_shoe_quaternion)
                    # norm per sensor
                    cur_shoe_acc     = scale(cur_shoe_acc)
                    cur_shoe_gyro    = scale(cur_shoe_gyro)
                    
                    cur_shoe_data    = np.concatenate((cur_shoe_acc, cur_shoe_gyro), axis=1)
                
                cur_trail_frag = np.concatenate((cur_trail_frag, cur_shoe_data), axis=1)
            # append label/subject/trial IDs
            cur_trail_frag = np.concatenate((cur_trail_frag, cur_trail[:,SELEC_LABEL:(SELEC_LABEL+1)]), axis=1)
            cur_trail_frag = np.concatenate((cur_trail_frag, cur_trail[:,SUBJECT_ID:(SUBJECT_ID+1)]), axis=1)
            cur_trail_frag = np.concatenate((cur_trail_frag, cur_trail[:,RUN_ID:(RUN_ID+1)]), axis=1)
            
            for (label_id,label) in enumerate(LABELS):
                
                cur_label_frag = pd.DataFrame(cur_trail_frag[cur_trail_frag[:,-3] == label])
                
                if cur_label_frag.shape[0] == 0 or cur_label_frag.shape[0] < window_size:
                    continue
                
                # get the IMU data only
                cur_label_frag = pd.DataFrame(cur_label_frag.values[:, 0:(cur_label_frag.shape[1]-3)])
                
                cur_label_segments_mid = preprocess_signal(cur_label_frag, window_size, overlap)
                cur_label_segments_mid = np.array(cur_label_segments_mid)
                
                cur_label_y_labels     = [label] * len(cur_label_segments_mid)
                cur_label_sub_labels   = [sub_id] * len(cur_label_segments_mid)
                cur_label_trial_labels = [trail_id] * len(cur_label_segments_mid)
                
                # if label_id == 0:
                #     cur_label_segments = cur_label_segments_mid
                # else:
                #     cur_label_segments.extend(cur_label_segments_mid)
                
                # distinguish whether the current 'features' should be put into train set or test set, according to the predivided user info
                if (sub_id in TRAIN_SUBJECTS_ID) and (trail_id in TRAIN_SUBJECTS_TRIAL_ID):
                    if len(X_train) == 0:
                        X_train        = cur_label_segments_mid
                        Y_train        = cur_label_y_labels
                        User_ids_train = cur_label_sub_labels
                    else:
                        X_train        = np.vstack((X_train, cur_label_segments_mid))
                        Y_train        = Y_train + cur_label_y_labels
                        User_ids_train = User_ids_train + cur_label_sub_labels
                else:
                    if len(X_test) == 0:
                        X_test         = cur_label_segments_mid
                        Y_test         = cur_label_y_labels
                        User_ids_test  = cur_label_sub_labels
                    else:
                        X_test         = np.vstack((X_test, cur_label_segments_mid))
                        Y_test         = Y_test + cur_label_y_labels
                        User_ids_test  = User_ids_test + cur_label_sub_labels
    # X_train = np.swapaxes(X_train,1,2)
    # X_train = np.expand_dims(X_train, 1)
    # X_test  = np.swapaxes(X_test,1,2)
    # X_test  = np.expand_dims(X_test,  1)

    return X_train, X_test, Y_train, Y_test, User_ids_train, User_ids_test