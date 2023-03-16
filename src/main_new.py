from datetime import datetime
import json
from logging import basicConfig, getLogger, StreamHandler, DEBUG, WARNING
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from utils.constants import *
from utils.load_HAPT_dataset.load_HAPT_dataset import load_HAPT_raw_data
from utils.load_Opportunity_dataset.load_Opportunity_dataset import load_Opportunity_data
from utils.utils import *
import time

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory

# def load_raw_data(dataset_name, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID):
def load_raw_data(dataset_name):
    # load raw data
    if dataset_name == 'HAPT':
        DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID, TRAIN_SUBJECTS_ID,\
        TEST_SUBJECTS_ID, WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, POS_NUM,\
        separate_gravity_flag = get_HAPT_dataset_param(CUR_DIR,dataset_name)
        
        X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
            label2act, act2label = load_HAPT_raw_data(DATA_DIR, TRAIN_SUBJECTS_ID, ActID,
                                                      WINDOW_SIZE, OVERLAP,separate_gravity_flag,
                                                      cal_attitude_angle)
    
    elif dataset_name == 'Opportunity':
        DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS, TRIALS, SELEC_LABEL,\
           ACT_LABELS, ActID, TRAIN_SUBJECTS_ID, TRAIN_SUBJECTS_TRIAL_ID, WINDOW_SIZE, OVERLAP,\
           INPUT_CHANNEL, POS_NUM, separate_gravity_flag, to_NED_flag = get_Opportunity_dataset_param(CUR_DIR, dataset_name)
        
        X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
            label2act, act2label = load_Opportunity_data(DATA_DIR, SUBJECTS, TRIALS, SELEC_LABEL,
                                                         TRAIN_SUBJECTS_ID, TRAIN_SUBJECTS_TRIAL_ID,
                                                         ACT_LABELS, ActID, WINDOW_SIZE, OVERLAP,
                                                         separate_gravity_flag, cal_attitude_angle,
                                                         to_NED_flag)
        
        TEST_SUBJECTS_ID         = list(set(SUBJECTS) ^ set(TRAIN_SUBJECTS_ID))
    
    return X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
           label2act, act2label,\
           ACT_LABELS, ActID, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID,\
           MODELS_COMP_LOG_DIR, INPUT_CHANNEL, POS_NUM

def create_cuda_classifier(dataset_name, classifier_name, INPUT_CHANNEL, POS_NUM, data_length, nb_classes):
    
    classifier, classifier_func = create_classifier(dataset_name, classifier_name, INPUT_CHANNEL, POS_NUM,
                                                    data_length, nb_classes)
    if INFERENCE_DEVICE == 'TEST_CUDA':
        classifier.cuda() # 这变了
    print(classifier)
    classifier_parameter = get_parameter_number(classifier)
    
    return classifier, classifier_func, classifier_parameter

for dataset_name in DATASETS:
    
    X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
        label2act, act2label,\
        ACT_LABELS, ActID, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID,\
        MODELS_COMP_LOG_DIR, INPUT_CHANNEL, POS_NUM = load_raw_data(dataset_name)
    
    All_data   = np.concatenate((X_train, X_test), axis = 0)
    All_labels = np.concatenate((y_train, y_test))
    All_users  = np.concatenate((User_ids_train, User_ids_test))
    ALL_SUBJECTS_ID = list(set(TRAIN_SUBJECTS_ID + TEST_SUBJECTS_ID))
    
    for classifier_name in CLASSIFIERS:
        
        # set logging settings
        EXEC_TIME, LOG_DIR, MODEL_DIR, logger = logging_settings(classifier_name, CUR_DIR, dataset_name)
        time2 = 0
        
        for subject_id in ALL_SUBJECTS_ID:
            
            # get train and test subjects
            TEST_SUBJECTS_ID = [subject_id]
            TRAIN_SUBJECTS_ID = list(set(ALL_SUBJECTS_ID).difference(set(TEST_SUBJECTS_ID)))
            
            ################### LOAD DATA AND DATA PREPROCESSING ##################
            start1 = time.time()
            
            # # cal_attitude_angle is only for proposed algorithms
            # if classifier_name in ['Deep_Conv_LSTM_torch','Deep_attn_Conv_LSTM_torch','DeepSense_torch','AttnSense_torch','Deep_Attention_Transformer_torch','Comple_filter_Conv_Transformer_torch']:
            #     cal_attitude_angle = False
            # else:
            #     cal_attitude_angle = True
            
            # # load raw data and important dataset param
            # X_train, X_test, y_train, y_test, label2act, act2label,\
            #     ACT_LABELS, ActID, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID,\
            #     MODELS_COMP_LOG_DIR, INPUT_CHANNEL = load_raw_data(dataset_name,TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID)
            
            X_train, y_train = get_loso_train_test_data(TRAIN_SUBJECTS_ID, All_users, All_data, All_labels)
            X_test, y_test   = get_loso_train_test_data(TEST_SUBJECTS_ID, All_users, All_data, All_labels)
            
            if classifier_name in ['Comple_Filter_Conv_Transformer_torch']:
                for (pos_id,pos) in enumerate(range(POS_NUM)):
                    if pos_id == 0:
                        X_train_cat = complementary_filter(X_train[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], COMPLEMAENTARY_ALPHA)
                        X_test_cat  = complementary_filter(X_test[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], COMPLEMAENTARY_ALPHA)
                    else:
                        X_train_mid = complementary_filter(X_train[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], COMPLEMAENTARY_ALPHA)
                        X_test_mid  = complementary_filter(X_test[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], COMPLEMAENTARY_ALPHA)
                        X_train_cat = np.concatenate((X_train_cat, X_train_mid), axis = 2)
                        X_test_cat  = np.concatenate((X_test_cat, X_test_mid), axis = 2)
                X_train             = X_train_cat
                X_test              = X_test_cat
            
            if classifier_name in ['DeepSense_torch','AttnSense_torch','GlobalFusion_torch']:
                for (pos_id,pos) in enumerate(range(POS_NUM)):
                    if pos_id == 0:
                        X_train_cat = STFT_transform(X_train[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], y_train, STFT_intervals, test_split)
                        X_test_cat  = STFT_transform(X_test[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], y_test, STFT_intervals, test_split)
                    else:
                        X_train_mid = STFT_transform(X_train[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], y_train, STFT_intervals, test_split)
                        X_test_mid  = STFT_transform(X_test[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], y_test, STFT_intervals, test_split)
                        X_train_cat = np.concatenate((X_train_cat, X_train_mid), axis = 1)
                        X_test_cat  = np.concatenate((X_test_cat, X_test_mid), axis = 1)
                X_train             = X_train_cat
                X_test              = X_test_cat
            # X_train=np.expand_dims(X_train[:,0,:,:], axis=1)
            # X_test=np.expand_dims(X_test[:,0,:,:], axis=1)
            end1 = time.time()
            #######################################################################
            
            ############### LOG DATASET INFO AND NETWORK PARAMETERS ###############
            # log the information of datasets
            log_dataset_info(logger, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, X_train, X_test,
                             y_train, y_test, cal_attitude_angle, ACT_LABELS, ActID)
            
            # check the category imbalance
            nb_classes = len(np.unique(y_train))
            check_class_balance(logger, y_train.flatten(), y_test.flatten(),
                                label2act=label2act, n_class=nb_classes)
            
            # log the hyper-parameters
            log_HyperParameters(logger, BATCH_SIZE, EPOCH, LR, len(ALL_SUBJECTS_ID))
            #######################################################################
            
            ##### SPLIT TRAINSET TO TRAIN AND VAL DATASETS #####
            # Split data by preserving the percentage of samples for each class.
            # here shuffle=False due to trainset has been shuffled through shuffle_trainset func
            if subject_id == min(ALL_SUBJECTS_ID):
                models, scores, log_training_duration = initialize_saving_variables(X_train, X_test, nb_classes, len(ALL_SUBJECTS_ID))
                
            ############ read train and val dataset ############
            # data and classifier preparation
            X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, y_train,
                                                        test_size=0.1,
                                                        random_state=6,
                                                        stratify=y_train)
            ####################################################
            aa = X_val.squeeze()
            # create classifier, put it into cuda and get the parameter number of it
            classifier, classifier_func, classifier_parameter = create_cuda_classifier(dataset_name, classifier_name,
                                                                                       INPUT_CHANNEL, POS_NUM, X_train.shape[-1],
                                                                                       nb_classes)
            
            ###### TRAIN FOR EACH FOLD, if trained, print 'Already_done' ######
            # log train, validation and test dataset info, log the network
            log_redivdataset_network_info(logger, subject_id, X_tr, X_val, X_test, Y_tr, Y_val,
                                          y_test, nb_classes, classifier_parameter, classifier)
            
            # train the network and save the best validation model
            output_directory_models = MODEL_DIR+'\SUBJECT_'+str(subject_id)+'\\'
            flag_output_directory_models = create_directory(output_directory_models)
            if PATTERN == 'TRAIN':
                flag_output_directory_models = PATTERN
            if flag_output_directory_models is not None:
                # for each subject, train the network and save the best validation model
                print('SUBJECT_'+str(subject_id)+': start to train')
                history, per_training_duration, log_training_duration = classifier_func.train_op(classifier, EPOCH, BATCH_SIZE, LR,
                                                                                                 X_tr, Y_tr, X_val, Y_val, X_test, y_test,
                                                                                                 output_directory_models, log_training_duration,
                                                                                                 test_split)
            else:
                print('Already_done: '+'SUBJECT_'+str(subject_id))
                # read the training duration of current subject
                per_training_duration = pd.read_csv(output_directory_models+'score.csv',
                                                    skiprows=1, nrows=1, header = None)[1][0]
                log_training_duration.append(per_training_duration)
            ###################################################################
            
            ########## TEST FOR EACH SUBJECT, record inference time ###########
            # input: X_tr, X_val, X_test, output: pred_train, pred_valid, pred_test (the one hot predictions)
            # save the metrics per subject
            pred_train, pred_valid, pred_test, scores, time_duration = classifier_func.predict_tr_val_test(classifier, nb_classes, ACT_LABELS,
            # pred_train, pred_valid, pred_test, scores = classifier_func.predict_tr_val_test(classifier, nb_classes, ACT_LABELS,
                                                                                            X_tr, X_val, X_test,
                                                                                            Y_tr, Y_val, y_test,
                                                                                            scores, per_training_duration,
                                                                                            subject_id, output_directory_models,
                                                                                            test_split)
            
            time2 = time2 + time_duration
            ###################################################################
            
            #######################################################################
            
        ################ LOG TEST RESULTS, LOG INFERENCE TIME #################
        # Log the Test Scores of every subject
        log_every_SUBJECT_score(logger, log_training_duration, scores, label2act, nb_classes, len(ALL_SUBJECTS_ID))
        # Log the averaged Score of different subjects
        log_averaged_SUBJECT_scores(logger, log_training_duration, scores, label2act, nb_classes, len(ALL_SUBJECTS_ID))
        
        # Log the inference time including data loading, preprocessing and model inference
        preprocess_time, inference_time = log_inference_time(start1, end1, y_train, y_test, time2,
                                                             dataset_name, classifier_name, len(ALL_SUBJECTS_ID), logger)
        
        save_classifiers_comparison(MODELS_COMP_LOG_DIR, CLASSIFIERS, classifier_name,
                                    scores, ALL_SUBJECTS_ID, preprocess_time, inference_time,
                                    len(ALL_SUBJECTS_ID))
        #######################################################################
    
    a = 1