"""Preprocess raw data for creating input for DL tasks."""
import glob
import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.load_HAPT_dataset.preprocessing import Preprocess

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
        signal = StandardScaler().fit_transform(signal)
        return pd.DataFrame(signal, columns=["x", "y", "z"])
    elif scaler == "minmax":
        signal = MinMaxScaler(feature_range=minmax_range).fit_transform(signal)
        return pd.DataFrame(signal, columns=["x", "y", "z"])


def preprocess_raw_data(DATA_DIR, TRAIN_SUBJECTS, ActID, window_size, overlap, cal_attitude_angle,
                        scaler, separate_gravity_flag=False):

    acc_files  = sorted(glob.glob(os.path.join(DATA_DIR, "RawData", "acc*.txt")))
    gyro_files = sorted(glob.glob(os.path.join(DATA_DIR, "RawData", "gyro*.txt")))
    label_info = pd.read_table(
        os.path.join(DATA_DIR, "RawData", "labels.txt"),
        sep=" ",
        header=None,
        names=["ExpID", "UserID", "ActID", "ActStart", "ActEnd"],
    )

    X_train = np.array([])
    Y_train = np.array([])
    X_test  = np.array([])
    Y_test  = np.array([])

    for acc_file, gyro_file in zip(acc_files, gyro_files):
        exp_id  = int(acc_file.split("exp")[1][:2])
        user_id = int(acc_file.split("user")[1][:2])

        temp_label_info = label_info[
            (label_info.ExpID == exp_id)
            & (label_info.UserID == user_id)
            & (label_info.ActID.isin(ActID))
        ]

        acc_raw  = pd.read_table(acc_file, sep=" ", header=None, names=["x", "y", "z"])
        gyro_raw = pd.read_table(gyro_file, sep=" ", header=None, names=["x", "y", "z"])
        
        # per axis(x,y,z) does Z-normalization
        if separate_gravity_flag:
            pre_obj = Preprocess()
            acc_body, acc_grav = pre_obj.separate_gravity(acc_raw, cal_attitude_angle) # seperate gravity from acc
            acc_body = scale(acc_body, scaler=scaler)
            acc_grav = scale(acc_grav, scaler=scaler)
            # if cal_attitude_angle:
            #     acc_grav_amplitude = acc_grav.apply(lambda x : x**2).sum(1).apply(lambda x : x**0.5)
            #     acc_grav = acc_grav.div(acc_grav_amplitude, axis=0)
            gyro_raw = scale(gyro_raw, scaler=scaler)
        else:
            acc_raw = scale(acc_raw, scaler=scaler)
            gyro_raw = scale(gyro_raw, scaler=scaler)

        for _, _, act_id, act_start, act_end in temp_label_info.values:
            # filter noises and extract windows
            if separate_gravity_flag:
                temp_acc_body = acc_body.iloc[act_start : act_end + 1]
                temp_acc_grav = acc_grav.iloc[act_start : act_end + 1]
                temp_gyro_raw = gyro_raw.iloc[act_start : act_end + 1]
                tGravityAccXYZ = preprocess_signal(temp_acc_grav, window_size, overlap)
                tBodyGyroXYZ   = preprocess_signal(temp_gyro_raw, window_size, overlap)
                tBodyAccXYZ    = preprocess_signal(temp_acc_body, window_size, overlap)
                channel_num = 9
                column_ind = ["GravAccX", "GravAccY", "GravAccZ",
                              "GyroX",    "GyroY",    "GyroZ",
                              "BodyAccX", "BodyAccY", "BodyAccZ"]
            else:
                temp_acc_raw = acc_raw.iloc[act_start : act_end + 1]
                temp_gyro_raw = gyro_raw.iloc[act_start : act_end + 1]
                tAccXYZ = preprocess_signal(temp_acc_raw, window_size, overlap)
                tBodyGyroXYZ = preprocess_signal(temp_gyro_raw, window_size, overlap)
                channel_num = 6
                column_ind = ["AccX",  "AccY",  "AccZ",
                              "GyroX", "GyroY", "GyroZ"]
            # the number of windows in current experiment, user and action, put them into features
            features = np.zeros((len(tBodyGyroXYZ), window_size, channel_num)) # len(tAccXYZ):number of windows, 128:window size, 6:the axises of channels
            # concatenate acc and gyro data into 'feature', then the seg windows of data are put into features
            for i in range(len(tBodyGyroXYZ)):
                if separate_gravity_flag:
                    np_concat = np.concatenate((tGravityAccXYZ[i], tBodyGyroXYZ[i], tBodyAccXYZ[i]), 1)
                else:
                    np_concat = np.concatenate((tAccXYZ[i], tBodyGyroXYZ[i]), 1)
                feature = pd.DataFrame(
                    np_concat,
                    columns=column_ind,
                )
                features[i] = feature
            
            # Record the y_labels
            y_labels = [act_id] * len(tBodyGyroXYZ)
            
            # distinguish whether the current 'features' should be put into train set or test set, according to the predivided user info
            if user_id in TRAIN_SUBJECTS:
                if len(X_train) == 0:
                    X_train = features
                    Y_train = y_labels
                    User_ids_train = [user_id] * len(y_labels)
                else:
                    X_train = np.vstack((X_train, features))
                    Y_train = Y_train + y_labels
                    User_ids_train = User_ids_train + [user_id] * len(y_labels)
            else:
                if len(X_test) == 0:
                    X_test = features
                    Y_test = y_labels
                    User_ids_test = [user_id] * len(y_labels)
                else:
                    X_test = np.vstack((X_test, features))
                    Y_test = Y_test + y_labels
                    User_ids_test = User_ids_test + [user_id] * len(y_labels)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], channel_num, 1)
    X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], channel_num, 1)

    return X_train, X_test, Y_train, Y_test, User_ids_train, User_ids_test