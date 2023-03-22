import os
# import sys
# from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.constants import *
from utils.utils import *
import time

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args

        for dataset_name in [args.DATASETS]:
            
            # get the raw data for loso
            label2act, ACT_LABELS, ActID, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID,\
                   MODELS_COMP_LOG_DIR, INPUT_CHANNEL, POS_NUM,\
                   All_data, All_labels, All_users, ALL_SUBJECTS_ID = get_raw_data(args, dataset_name, CUR_DIR)
            
            for classifier_name in [args.CLASSIFIERS]:
                
                # logging settings
                EXEC_TIME, LOG_DIR, MODEL_DIR, logger = logging_settings(classifier_name, CUR_DIR, dataset_name)
                
                for subject_id in ALL_SUBJECTS_ID:
                    
                    # get training and test subjects
                    TEST_SUBJECTS_ID = [subject_id]
                    TRAIN_SUBJECTS_ID = list(set(ALL_SUBJECTS_ID).difference(set(TEST_SUBJECTS_ID)))
                    
                    ############### OBTAIN TRAINING AND TEST DATA FOR LOSO ################
                    X_train, y_train = get_loso_train_test_data(TRAIN_SUBJECTS_ID, All_users, All_data, All_labels)
                    X_test,  y_test  = get_loso_train_test_data(TEST_SUBJECTS_ID, All_users, All_data, All_labels)
                    #############################################################################
                    
                    ############### LOG DATASET INFO AND NETWORK PARAMETERS ###############
                    nb_classes = log_dataset_training_info(logger, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, X_train, X_test,
                                                           y_train, y_test, args.cal_attitude_angle, ACT_LABELS, ActID,
                                                           args.BATCH_SIZE, args.EPOCH, args.LR, ALL_SUBJECTS_ID, label2act)
                    #######################################################################
                    
                    ############ SPLIT TRAINING DATASET TO TRAIN AND VAL DATASETS ############
                    # initialize evaluation variables of a dict
                    if subject_id == min(ALL_SUBJECTS_ID):
                        models, scores, log_training_duration = initialize_saving_variables(X_train, X_test, nb_classes, len(ALL_SUBJECTS_ID))
                    # split training and val datasets
                    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, y_train,
                                                                test_size=0.1,
                                                                random_state=6,
                                                                stratify=y_train)
                    ##################################################################
                    
                    ################ CREATE CLASSIFIER OBJECT, GET PARAMATER NUM ################
                    classifier, classifier_func, classifier_parameter = create_cuda_classifier(dataset_name, classifier_name,
                                                                                               INPUT_CHANNEL, POS_NUM, X_train.shape[-1],
                                                                                               nb_classes)
                    ################ START TRAINING FOR EACH SUBJECT ########################
                    # log train, validation and test dataset info, log the network
                    log_redivdataset_network_info(logger, subject_id, X_tr, X_val, X_test, Y_tr, Y_val,
                                                  y_test, nb_classes, classifier_parameter, classifier)
                    # train the network and save the best validation model
                    output_directory_models, flag_output_directory_models = create_model_direc(MODEL_DIR, args.PATTERN, subject_id)
                    
                    if flag_output_directory_models is not None:
                        # for each subject, train the network and save the best validation model
                        print('SUBJECT_'+str(subject_id)+': start to train')
                        history, per_training_duration, log_training_duration = classifier_func.train_op(classifier, args.EPOCH, args.BATCH_SIZE, args.LR,
                                                                                                         X_tr, Y_tr, X_val, Y_val, X_test, y_test,
                                                                                                         output_directory_models, log_training_duration,
                                                                                                         args.test_split)
                    else:
                        print('Already_done: '+'SUBJECT_'+str(subject_id))
                        # read the training duration of current subject
                        per_training_duration = pd.read_csv(output_directory_models+'score.csv',
                                                            skiprows=1, nrows=1, header = None)[1][0]
                        log_training_duration.append(per_training_duration)
                    ###################################################################
                    
                    ########## TEST FOR EACH SUBJECT, record inference time ###########
                    # input: X_tr, X_val, X_test, output: pred_train, pred_valid, pred_test (the one hot predictions)
                    pred_train, pred_valid, pred_test, scores = classifier_func.predict_tr_val_test(classifier, nb_classes, ACT_LABELS,
                                                                                                    X_tr, X_val, X_test,
                                                                                                    Y_tr, Y_val, y_test,
                                                                                                    scores, per_training_duration,
                                                                                                    subject_id, output_directory_models,
                                                                                                    args.test_split)
                    ###################################################################
                
                ################## LOG TEST RESULTS, SAVE TO TABLES ###################
                log_test_results(logger, log_training_duration, scores, label2act, nb_classes, ALL_SUBJECTS_ID,
                                 MODELS_COMP_LOG_DIR, [args.CLASSIFIERS], classifier_name)
                #######################################################################
                

def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)