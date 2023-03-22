"""Collection of utility functions"""
from datetime import datetime
from logging import basicConfig, getLogger, Formatter, FileHandler, StreamHandler, DEBUG, WARNING
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
# from utils.constants import INFERENCE_DEVICE
from utils.constants import parse_args
args = parse_args()
INFERENCE_DEVICE = args.INFERENCE_DEVICE

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
import torch.nn.functional as F
from utils.constants import *
from utils.load_HAPT_dataset.load_HAPT_dataset import load_HAPT_raw_data
from utils.load_Opportunity_dataset.load_Opportunity_dataset import load_Opportunity_data

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

shap.initjs()
logger = getLogger(__name__)

# load raw training and testing data of each dataset
def load_raw_data(args, dataset_name, CUR_DIR):
    # load raw data
    if dataset_name == 'HAPT':
        DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID, TRAIN_SUBJECTS_ID,\
        TEST_SUBJECTS_ID, WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, POS_NUM,\
        separate_gravity_flag = get_HAPT_dataset_param(CUR_DIR,dataset_name)
        
        X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
            label2act, act2label = load_HAPT_raw_data(DATA_DIR, TRAIN_SUBJECTS_ID, ActID,
                                                      WINDOW_SIZE, OVERLAP,separate_gravity_flag,
                                                      args.cal_attitude_angle)
    
    elif dataset_name == 'Opportunity':
        DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS, TRIALS, SELEC_LABEL,\
           ACT_LABELS, ActID, TRAIN_SUBJECTS_ID, TRAIN_SUBJECTS_TRIAL_ID, WINDOW_SIZE, OVERLAP,\
           INPUT_CHANNEL, POS_NUM, separate_gravity_flag, to_NED_flag = get_Opportunity_dataset_param(CUR_DIR, dataset_name)
        
        X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
            label2act, act2label = load_Opportunity_data(DATA_DIR, SUBJECTS, TRIALS, SELEC_LABEL,
                                                         TRAIN_SUBJECTS_ID, TRAIN_SUBJECTS_TRIAL_ID,
                                                         ACT_LABELS, ActID, WINDOW_SIZE, OVERLAP,
                                                         separate_gravity_flag, args.cal_attitude_angle,
                                                         to_NED_flag)
        
        TEST_SUBJECTS_ID         = list(set(SUBJECTS) ^ set(TRAIN_SUBJECTS_ID))
    
    return X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
           label2act, act2label,\
           ACT_LABELS, ActID, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID,\
           MODELS_COMP_LOG_DIR, INPUT_CHANNEL, POS_NUM

# get the raw data for loso
def get_raw_data(args, dataset_name, CUR_DIR):
    
    X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
        label2act, act2label,\
        ACT_LABELS, ActID, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID,\
        MODELS_COMP_LOG_DIR, INPUT_CHANNEL, POS_NUM = load_raw_data(args, dataset_name, CUR_DIR)
    
    All_data   = np.concatenate((X_train, X_test), axis = 0)
    All_labels = np.concatenate((y_train, y_test))
    All_users  = np.concatenate((User_ids_train, User_ids_test))
    ALL_SUBJECTS_ID = list(set(TRAIN_SUBJECTS_ID + TEST_SUBJECTS_ID))
    
    return label2act, ACT_LABELS, ActID, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID,\
           MODELS_COMP_LOG_DIR, INPUT_CHANNEL, POS_NUM, All_data, All_labels,\
           All_users, ALL_SUBJECTS_ID

def shuffle_trainset(X_train, y_train):
    
    indices = np.arange(X_train.shape[0])
    np.random.seed(66)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    return X_train, y_train

def shuffle_train_test(X_train, y_train, X_test, y_test):
    
    x_dataset = np.concatenate((X_train, X_test), axis=0)
    y_dataset = np.concatenate((y_train, y_test), axis=0)
    indices = np.arange(x_dataset.shape[0])
    np.random.seed(66)
    np.random.shuffle(indices)
    x_dataset = x_dataset[indices]
    bb = x_dataset.squeeze()
    y_dataset = y_dataset[indices]
    X_train = x_dataset[:len(y_train),:,:,:]
    y_train = y_dataset[:len(y_train)]
    X_test  = x_dataset[len(y_train):,:,:,:]
    y_test  = y_dataset[len(y_train):]
    
    return X_train, y_train, X_test, y_test

def check_class_balance(logger, y_train: np.ndarray, y_test: np.ndarray,
                        label2act: Dict[int, str], n_class: int = 12
) -> None:
    c_train = Counter(y_train)
    c_test = Counter(y_test)

    for c, mode in zip([c_train, c_test], ["train", "test"]):
        logger.debug(f"{mode} labels")
        len_y = sum(c.values())
        for label_id in range(n_class):
            logger.debug(
                f"{label2act[label_id]} ({label_id}): {c[label_id]} samples ({c[label_id] / len_y * 100:.04} %)"
            )

def create_directory(directory_path): 
    if os.path.exists(directory_path): 
        return None
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            # in case another machine created the path meanwhile !:(
            return None 
        return directory_path

# create the directory of the training model
def create_model_direc(MODEL_DIR, PATTERN, subject_id):
    
    output_directory_models = MODEL_DIR+'\SUBJECT_'+str(subject_id)+'\\'
    flag_output_directory_models = create_directory(output_directory_models)
    if PATTERN == 'TRAIN':
        flag_output_directory_models = PATTERN
    
    return output_directory_models, flag_output_directory_models

# Logging settings
def logging_settings(classifier_name, CUR_DIR, dataset_name):
    
    EXEC_TIME = classifier_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = os.path.join(CUR_DIR, f"logs", dataset_name, classifier_name, f"{EXEC_TIME}")
    MODEL_DIR = os.path.join(CUR_DIR, f"saved_model", dataset_name, classifier_name)
    create_directory(LOG_DIR) # Create log directory
    
    # create log object with classifier_name
    cur_classifier_log = getLogger(classifier_name)
    # set recording format
    formatter = Formatter("%(levelname)s: %(asctime)s: %(filename)s: %(funcName)s: %(message)s")
    # create FileHandler with current LOG_DIR and format
    fileHandler = FileHandler(f"{LOG_DIR}/{EXEC_TIME}.log")
    fileHandler.setFormatter(formatter)
    streamHandler = StreamHandler()
    streamHandler.setFormatter(formatter)

    mpl_logger = getLogger("matplotlib")  # Suppress matplotlib logging
    mpl_logger.setLevel(WARNING)
    
    cur_classifier_log.setLevel(DEBUG)
    cur_classifier_log.addHandler(fileHandler)
    cur_classifier_log.addHandler(streamHandler)
    
    # important! get current logger with its name (the name is set with classifier_name)
    logger = getLogger(classifier_name)
    logger.setLevel(DEBUG)
    logger.debug(f"{LOG_DIR}/{EXEC_TIME}.log")
    
    return EXEC_TIME, LOG_DIR, MODEL_DIR, logger

# obtain the training and test data for loso
def get_loso_train_test_data(SUBJECTS_IDS, All_users, All_data, All_labels):
    for (r_id, sub_id) in enumerate(SUBJECTS_IDS):
        if r_id == 0:
            ids = np.where(All_users == sub_id)[0]
        else:
            ids = np.concatenate((ids, np.where(All_users == sub_id)[0]))
    X = All_data[ids,:,:,:]
    y = All_labels[ids]
    return X, y

# initialize evaluation variables of a dict
def initialize_saving_variables(X_train, X_test, nb_classes, SUBJECT_NUM):
    
    # for test sets there are predictions for SUBJECT_NUM times
    models = []
    scores: Dict[str, Dict[str, List[Any]]] = {
        "logloss": {"train": [], "valid": [], "test": []},
        "accuracy": {"train": [], "valid": [], "test": []},
        "macro-precision": {"train": [], "valid": [], "test": []},
        "macro-recall": {"train": [], "valid": [], "test": []},
        "macro-f1": {"train": [], "valid": [], "test": []},
        "weighted-f1": {"train": [], "valid": [], "test": []},
        "micro-f1": {"train": [], "valid": [], "test": []},
        "per_class_f1": {"train": [], "valid": [], "test": []},
        "confusion_matrix": {"train": [], "valid": [], "test": []},
    }    
    log_training_duration = []
    
    return models, scores, log_training_duration

# obtain the network param number
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Network_Total_Parameters:', total_num, 'Network_Trainable_Parameters:', trainable_num)
    return {'Total': total_num, 'Trainable': trainable_num}

# create classifier object, get classifier param number
def create_cuda_classifier(dataset_name, classifier_name, INPUT_CHANNEL, POS_NUM, data_length, nb_classes):
    
    classifier, classifier_func = create_classifier(dataset_name, classifier_name, INPUT_CHANNEL, POS_NUM,
                                                    data_length, nb_classes)
    if INFERENCE_DEVICE == 'TEST_CUDA':
        classifier.cuda() # 这变了
    print(classifier)
    classifier_parameter = get_parameter_number(classifier)
    
    return classifier, classifier_func, classifier_parameter

# obtain the output of the network
def model_predict(net, x_data, y_data, test_split=1):
    predict = [] 
    output = []
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    data_loader = Data.DataLoader(dataset = torch_dataset,
                                  batch_size = x_data.shape[0] // test_split,
                                  shuffle = False)    
    for step, (x,y) in enumerate(data_loader):
        with torch.no_grad():
            if INFERENCE_DEVICE == 'TEST_CUDA':
                x = x.cuda() # 这变了
            output_bc = net(x)[0]
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
            out = output_bc.cpu().data.numpy()
            output.extend(out)
    return output

# obtain the test results of the training, val and test datasets
def predict_tr_val_test(network, nb_classes, LABELS,
                        train_x, val_x, test_x,
                        train_y, val_y, test_y,
                        scores, per_training_duration,
                        run_id, output_directory_models,
                        test_split):
    
    # generate network objects
    network_obj = network
    # load best saved validation models
    best_validation_model = output_directory_models+'best_validation_model.pkl'
    network_obj.load_state_dict(torch.load(best_validation_model))
    network_obj.eval()
    # get outputs of best saved validation models by concat them, input: train_x, val_x, test_x
    pred_train = np.array(model_predict(network_obj, train_x, train_y, test_split))
    pred_valid = np.array(model_predict(network_obj, val_x, val_y, test_split))
    pred_test = np.array(model_predict(network_obj, test_x, test_y, test_split))
    
    # record the metrics of each subject, initialize the score per CV
    score: Dict[str, Dict[str, List[Any]]] = {
                "logloss": {"train": [], "valid": [], "test": []},
                "accuracy": {"train": [], "valid": [], "test": []},
                "macro-precision": {"train": [], "valid": [], "test": []},
                "macro-recall": {"train": [], "valid": [], "test": []},
                "macro-f1": {"train": [], "valid": [], "test": []},
                "weighted-f1": {"train": [], "valid": [], "test": []},
                "micro-f1": {"train": [], "valid": [], "test": []},
                "per_class_f1": {"train": [], "valid": [], "test": []},
                "confusion_matrix": {"train": [], "valid": [], "test": []},
                }
    # loss_function = nn.CrossEntropyLoss(reduction='sum')
    loss_function = LabelSmoothingCrossEntropy()
    for pred, X, y, mode in zip(
        [pred_train, pred_valid, pred_test], [train_x, val_x, test_x], [train_y, val_y, test_y], ["train", "valid", "test"]
    ):
        loss, acc, macro_f1_val = get_test_loss_acc(network_obj, loss_function, X, y, test_split)
        pred = pred.argmax(axis=1)
        # y is already the argmaxed category
        scores["logloss"][mode].append(loss)
        scores["accuracy"][mode].append(acc)
        scores["macro-precision"][mode].append(precision_score(y, pred, average="macro"))
        scores["macro-recall"][mode].append(recall_score(y, pred, average="macro"))
        scores["macro-f1"][mode].append(f1_score(y, pred, average="macro"))
        scores["weighted-f1"][mode].append(f1_score(y, pred, average="weighted"))
        scores["micro-f1"][mode].append(f1_score(y, pred, average="micro"))
        scores["per_class_f1"][mode].append(f1_score(y, pred, average=None))
        scores["confusion_matrix"][mode].append(confusion_matrix(y, pred, normalize=None))
        
        # record the metrics of each subject
        score["logloss"][mode].append(loss)
        score["accuracy"][mode].append(acc)
        score["macro-precision"][mode].append(precision_score(y, pred, average="macro"))
        score["macro-recall"][mode].append(recall_score(y, pred, average="macro"))
        score["macro-f1"][mode].append(f1_score(y, pred, average="macro"))
        score["weighted-f1"][mode].append(f1_score(y, pred, average="weighted"))
        score["micro-f1"][mode].append(f1_score(y, pred, average="micro"))
        score["per_class_f1"][mode].append(f1_score(y, pred, average=None))
        score["confusion_matrix"][mode].append(confusion_matrix(y, pred, normalize=None))
    
    save_metrics_per_cv(score, per_training_duration,
                        run_id, nb_classes, LABELS,
                        test_y, pred_test,
                        output_directory_models)
    
    return pred_train, pred_valid, pred_test, scores

# callculate the loss, acc and F1-scores of the input data
def get_test_loss_acc(net, loss_function, x_data, y_data, test_split=1):
    loss_sum_data = torch.tensor(0)
    true_sum_data = torch.tensor(0)
    output = []
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    data_loader = Data.DataLoader(dataset = torch_dataset,
                                  batch_size = x_data.shape[0] // test_split,
                                  shuffle = False)
    for step, (x,y) in enumerate(data_loader):
        with torch.no_grad():
            if INFERENCE_DEVICE == 'TEST_CUDA':
                x = x.cuda() # 这变了
                y = y.cuda() # 这变了
            output_bc = net(x)[0]
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
                
            out = output_bc.cpu().data.numpy()
            
            if INFERENCE_DEVICE == 'TEST_CUDA':
                pred_bc = torch.max(output_bc, 1)[1].data.cuda().squeeze() # 这变了
            else:
                pred_bc = torch.max(output_bc, 1)[1].data.squeeze()
            loss_bc = loss_function(output_bc, y)
            true_num_bc = torch.sum(pred_bc == y).data
            loss_sum_data = loss_sum_data + loss_bc
            true_sum_data = true_sum_data + true_num_bc
            
            output.extend(out)
    
    loss = loss_sum_data.data.item()/y_data.shape[0]
    acc = true_sum_data.data.item()/y_data.shape[0]
    
    output = np.array(output).argmax(axis=1)
    macro_f1 = f1_score(y_data, output, average="macro")
    
    return loss, acc, macro_f1

# save the best val model
def save_models(net, output_directory_models, 
                loss_train, loss_train_results, 
                accuracy_validation, accuracy_validation_results, 
                start_time, training_duration_logs):   
    
    output_directory_best_val = output_directory_models+'best_validation_model.pkl'         
    if accuracy_validation >= max(accuracy_validation_results):        
        torch.save(net.state_dict(), output_directory_best_val)

# log the test results in the training process
def log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
                loss_validation_results, accuracy_validation_results, loss_test_results, accuracy_test_results,
                output_directory_models):
    
    history = pd.DataFrame(data = np.zeros((EPOCH,7),dtype=np.float), 
                           columns=['train_acc','train_loss','val_acc','val_loss',
                                    'test_acc','test_loss','lr'])
    history['train_acc'] = accuracy_train_results
    history['train_loss'] = loss_train_results
    history['val_acc'] = accuracy_validation_results
    history['val_loss'] = loss_validation_results
    history['test_acc'] = accuracy_test_results
    history['test_loss'] = loss_test_results
    history['lr'] = lr_results
    
    # load saved models, predict, cal metrics and save logs    
    history.to_csv(output_directory_models+'history.csv', index=False)
    
    return history

# plot the history results of the training process
def plot_learning_history(EPOCH, history, path):
    """Plot learning curve
    Args:
        fit (Any): History object
        path (str, default="history.png")
    """
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
    axL.plot(history["train_loss"], label="train")
    axL.plot(history["val_loss"], label="validation")
    axL.plot(history["test_loss"], label="test")
    axL.set_title("Loss")
    axL.set_xlabel("epoch")
    axL.set_ylabel("loss")
    axL.legend(loc="upper right")

    axR.plot(history["train_acc"], label="train")
    axR.plot(history["val_acc"], label="validation")
    axR.plot(history["test_acc"], label="test")
    axR.set_title("Accuracy")
    axR.set_xlabel("epoch")
    axR.set_ylabel("accuracy")
    axR.legend(loc="upper right")

    fig.savefig(path+'history.png')
    plt.close()

# save the test results of each loso subject
def save_metrics_per_cv(score, per_training_duration,
                        subject_id, nb_classes, LABELS,
                        y_true, y_pred,
                        output_directory_models):
    
    # save training time
    per_training_duration_pd = pd.DataFrame(data = per_training_duration,
                                            index = ["training duration"],
                                            columns = ["SUBJECT_"+str(subject_id)])
    per_training_duration_pd.to_csv(output_directory_models+'score.csv', index=True)
    
    # save "logloss", "accuracy", "precision", "recall", "f1"
    score_pd = pd.DataFrame(data = np.zeros((7,3),dtype=np.float),
                            index=["logloss", "accuracy", "macro-precision", "macro-recall",
                                   "macro-f1", "weighted-f1", "micro-f1"], 
                            columns=["train", "valid", "test"])
    for row in score_pd.index:
        for column in score_pd.columns:
            score_pd.loc[row, column] = score[row][column][0]
    score_pd.to_csv(output_directory_models+'score.csv', index=True, mode='a+')
    
    # save "per_class_f1"
    pd.DataFrame(["per_class_f1"]).to_csv(output_directory_models+'score.csv', index=False, header=False, mode='a+')
    per_class_f1_pd = pd.DataFrame(data = np.zeros((3, nb_classes),dtype=np.str_),
                            index=["train", "valid", "test"], columns=LABELS)
    for row in per_class_f1_pd.index:
        flag = 0
        if row == "test":
            for (i, column) in enumerate(per_class_f1_pd.columns):
                if i in list(np.unique(y_true)):
                    per_class_f1_pd.loc[row, column] = np.str_(score["per_class_f1"][row][0][flag])
                    flag = flag + 1
                else:
                    per_class_f1_pd.loc[row, column] = 'missed_category'
        else:
            for (i, column) in enumerate(per_class_f1_pd.columns):
                per_class_f1_pd.loc[row, column] = score["per_class_f1"][row][0][i]
            
    per_class_f1_pd.to_csv(output_directory_models+'score.csv', index=True, mode='a+')
    
    # save confusion_matrix
    for key in score['confusion_matrix'].keys():
        pd.DataFrame(["confusion_matrix_"+key]).to_csv(output_directory_models+'score.csv', index=False, header=False, mode='a+')
        each_confusion_matrix = pd.DataFrame(data = np.zeros((nb_classes, nb_classes),dtype=np.float), 
                                             index = LABELS, columns=LABELS)
        # if missing categories exist
        if key == 'test':
            # two loops, one for row and one for column
            flag_cfm_row = 0
            # row loop
            for (i,row) in enumerate(each_confusion_matrix.index):
                if i in list(np.unique(y_true)):
                    flag_cfm_col = 0
                    # column loop
                    for (j,column) in enumerate(each_confusion_matrix.columns):
                        if j in list(np.unique(y_true)):
                            each_confusion_matrix.loc[row, column] = score['confusion_matrix'][key][0][flag_cfm_row][flag_cfm_col]
                            flag_cfm_col = flag_cfm_col + 1
                        else:
                            each_confusion_matrix.loc[row, column] = 'missed_category'
                    flag_cfm_row = flag_cfm_row + 1
                else:
                    for (j,column) in enumerate(each_confusion_matrix.columns):
                        each_confusion_matrix.loc[row, column] = 'missed_category'
        else:
            for (i,row) in enumerate(each_confusion_matrix.index):
                for (j,column) in enumerate(each_confusion_matrix.columns):
                    each_confusion_matrix.loc[row, column] = score['confusion_matrix'][key][0][i][j]
        each_confusion_matrix.to_csv(output_directory_models+'score.csv', index=True, mode='a+')
    
    # save the indexes of the false predictions
    y_pred = y_pred.argmax(axis=1)
    false_index = np.where(np.array(y_true)!=np.array(y_pred))[0].tolist()
    y_correct = np.array(y_true)[np.array(y_true)!=np.array(y_pred)].tolist()
    pre_false = np.array(y_pred)[np.array(y_true)!=np.array(y_pred)].tolist()
    false_pres = pd.DataFrame(data = np.zeros((len(false_index),3),dtype=np.int64), 
                              columns=['index','real_category','predicted_category'])
    false_pres['index'] = false_index
    false_pres['real_category'] = y_correct
    false_pres['predicted_category'] = pre_false
    false_pres.to_csv(output_directory_models+'score.csv', index=True, mode='a+')

# log the info of datasets and training process
def log_dataset_training_info(logger, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, X_train, X_test,
                              y_train, y_test, cal_attitude_angle, ACT_LABELS, ActID, BATCH_SIZE,
                              EPOCH, LR, ALL_SUBJECTS_ID, label2act):
    
    # log the information of datasets
    log_dataset_info(logger, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, X_train, X_test,
                     y_train, y_test, cal_attitude_angle, ACT_LABELS, ActID)
    
    # check the category imbalance
    nb_classes = len(np.unique(y_train))
    check_class_balance(logger, y_train.flatten(), y_test.flatten(),
                        label2act=label2act, n_class=nb_classes)
    
    # log the hyper-parameters
    log_HyperParameters(logger, BATCH_SIZE, EPOCH, LR, len(ALL_SUBJECTS_ID))
    
    return nb_classes

# log the info of datasets
def log_dataset_info(logger, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, X_train, X_test,
                     y_train, y_test, cal_attitude_angle, ACT_LABELS, ActID):
    logger.debug("---Dataset and preprocessing information---")
    logger.debug(f"TRAIN_SUBJECTS_ID = {TRAIN_SUBJECTS_ID}")
    logger.debug(f"TEST_SUBJECTS_ID = {TEST_SUBJECTS_ID}")
    logger.debug(f"X_train_shape = {X_train.shape}, X_test_shape={X_test.shape}")
    logger.debug(f"Y_train_shape = {y_train.shape}, Y_test.shape={y_test.shape}")
    logger.debug(f"Cal_Attitude_Angle = {cal_attitude_angle}")
    logger.debug(f"ACT_LABELS = {ACT_LABELS}")
    logger.debug(f"ActID = {ActID}")

# log the info of training hyperparameters
def log_HyperParameters(logger, BATCH_SIZE, EPOCH, LR, subject_num):
    logger.debug("---HyperParameters---")
    logger.debug(f"BATCH_SIZE : {BATCH_SIZE}, EPOCH : {EPOCH}, LR : {LR}, SUBJECT_NUM : {subject_num}")

# log the info of the dataset and the network
def log_redivdataset_network_info(logger, subject_id, X_tr, X_val, X_test, Y_tr, Y_val,
                                  y_test, nb_classes, classifier_parameter, classifier):
    if subject_id == 1:
        logger.debug("---Redivided dataset and network information---")
        logger.debug(f"X_train_shape={X_tr.shape}, X_validation_shape={X_val.shape}, X_test_shape={X_test.shape}")
        logger.debug(f"Y_train_shape={Y_tr.shape}, Y_validation_shape={Y_val.shape}, y_test_shape={y_test.shape}")
        logger.debug(f"num of categories = {nb_classes}")
        logger.debug(f"num of network parameter = {classifier_parameter}")
        logger.debug(f"the architecture of the network = {classifier}")

# log the testing results of each subject
def log_every_SUBJECT_score(logger, log_training_duration, scores, label2act, nb_classes, SUBJECT_NUM):
    for i in range(SUBJECT_NUM):
        # Log Every Subject Scores
        logger.debug("---Per Subject Scores, Subject"+str(i)+"---")
        
        # log per SUBJECT training time
        logger.debug(f"Training Duration = {log_training_duration[i]}s")
        
        for mode in ["train", "valid", "test"]:
            # log the average of "logloss", "accuracy", "precision", "recall", "f1"
            logger.debug(f"---{mode}---")
            logger.debug(f"logloss={round(scores['logloss'][mode][i],4)}, accuracy={round(scores['accuracy'][mode][i],4)},\
                         macro-precision={round(scores['macro-precision'][mode][i],4)},\
                         macro-recall={round(scores['macro-recall'][mode][i],4)},\
                         macro-f1={round(scores['macro-f1'][mode][i],4)},\
                         weighted-f1={round(scores['weighted-f1'][mode][i],4)},\
                             micro-f1={round(scores['micro-f1'][mode][i],4)}")

def log_averaged_SUBJECT_scores(logger, log_training_duration, scores, label2act, nb_classes, SUBJECT_NUM):
    # Log Averaged Score of all Subjects
    logger.debug("---Subject Averaged Scores---")    
    # log the average of training time
    logger.debug(f"Averaged Training Duration = {(np.mean(log_training_duration))}s")
    
    for mode in ["train", "valid", "test"]:
        
        # log the average of "logloss", "accuracy", "precision", "recall", "f1"
        logger.debug(f"---{mode}---")
        for metric in ["logloss", "accuracy", "macro-precision", "macro-recall", "macro-f1", "weighted-f1", "micro-f1"]:
            logger.debug(f"{metric}={round(np.mean(scores[metric][mode]),4)} +- {round(np.std(scores[metric][mode]),4)}")

# save test results to csv tables
def save_classifiers_comparison(MODELS_COMP_LOG_DIR, CLASSIFIERS, classifier_name,
                                scores, ALL_SUBJECTS_ID, SUBJECT_NUM):
    
    for i in range(len(CLASSIFIERS)):
        if i == 0:
            CLASSIFIERS_names = CLASSIFIERS[0]+'&'
        else:
            CLASSIFIERS_names = CLASSIFIERS_names+CLASSIFIERS[i]+'&'
    classifiers_comparison_log_dir = MODELS_COMP_LOG_DIR + CLASSIFIERS_names + '-comparison' + '.csv'
    
    # record Averaged_SUBJECT_scores
    averaged_score_pd = pd.DataFrame(data = np.zeros((6, 1),dtype=np.str_),
                                     index=["accuracy","macro-precision",
                                            "macro-recall","macro-f1",
                                            "weighted-f1", "micro-f1"], 
                                     columns=[classifier_name])
    for row in averaged_score_pd.index:
        for column in averaged_score_pd.columns:
            averaged_score_pd.loc[row][column] = np.str_(np.mean(scores[row]["test"])) + '+-' + np.str_(np.std(scores[row]["test"]))

    # record Every_SUBJECT_scores
    for subject_id in ALL_SUBJECTS_ID:
            # record Averaged_SUBJECT_scores
            persub_score_pd = pd.DataFrame(data = np.zeros((3, 1),dtype=np.str_),
                                           index=["sub_"+str(subject_id)+"_accuracy",
                                                  "sub_"+str(subject_id)+"_macro-f1",
                                                  "sub_"+str(subject_id)+"_weighted-f1"],
                                           columns=[classifier_name])
            for row in persub_score_pd.index:
                for column in persub_score_pd.columns:
                    persub_score_pd.loc[row][column] = np.str_(scores[row.replace("sub_"+str(subject_id)+"_", '')]["test"][subject_id-1])
            if subject_id == min(ALL_SUBJECTS_ID):
                persub_score_pd_concat = persub_score_pd
            else:
                persub_score_pd_concat = pd.concat([persub_score_pd_concat, persub_score_pd], axis=0)
            # persub_score_pd.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')
    
    if classifier_name == CLASSIFIERS[0]:
        if os.path.exists(classifiers_comparison_log_dir):
            os.remove(classifiers_comparison_log_dir)
        _ = create_directory(MODELS_COMP_LOG_DIR)
        
        # save Averaged_SUBJECT_scores to CSV
        pd.DataFrame(["Averaged_SUBJECT_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header = False, mode='a+')
        averaged_score_pd.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')
        
        # save Every_SUBJECT_scores to CSV
        pd.DataFrame(["Every_SUBJECT_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header = False, mode='a+')
        persub_score_pd_concat.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')

    else:
        # add averaged_scores of new classifier
        saved_averaged_scores  = pd.read_csv(classifiers_comparison_log_dir, skiprows=1, nrows=6, header=0, index_col=0)
        saved_averaged_scores  = pd.concat([saved_averaged_scores, averaged_score_pd], axis=1)
        # add every_subject_scores of new classifier
        saved_everysub_scores = pd.read_csv(classifiers_comparison_log_dir, skiprows=10, nrows=3*len(ALL_SUBJECTS_ID), header=0, index_col=0)
        saved_everysub_scores = pd.concat([saved_everysub_scores, persub_score_pd_concat], axis=1)
        
        os.remove(classifiers_comparison_log_dir)
        # _ = create_directory(MODELS_COMP_LOG_DIR)
        # save Averaged_SUBJECT_scores to CSV
        pd.DataFrame(["Averaged_SUBJECT_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header = False, mode='a+')
        saved_averaged_scores.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')
        # save Every_SUBJECT_scores to CSV
        pd.DataFrame(["Every_SUBJECT_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header = False, mode='a+')
        saved_everysub_scores.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')

# log the test scores, save to csv tables
def log_test_results(logger, log_training_duration, scores, label2act, nb_classes, ALL_SUBJECTS_ID,
                     MODELS_COMP_LOG_DIR, CLASSIFIERS, classifier_name):
    # Log the Test Scores of every subject
    log_every_SUBJECT_score(logger, log_training_duration, scores, label2act, nb_classes, len(ALL_SUBJECTS_ID))
    # Log the averaged Score of different subjects
    log_averaged_SUBJECT_scores(logger, log_training_duration, scores, label2act, nb_classes, len(ALL_SUBJECTS_ID))
    # save to tables
    save_classifiers_comparison(MODELS_COMP_LOG_DIR, CLASSIFIERS, classifier_name,
                                scores, ALL_SUBJECTS_ID, len(ALL_SUBJECTS_ID))

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence  = 1. - smoothing
        logprobs    = F.log_softmax(x, dim=-1)
        nll_loss    = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss    = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss        = confidence * nll_loss + smoothing * smooth_loss
        # return loss.mean()
        return loss.sum()
    
def complementary_filter(X, alpha):
    
    X       = X.squeeze()
    x_grav  = X[:,0:3,:]
    x_gyro  = X[:,3:6,:]
    x_acc   = X[:,6:9,:]
    
    attitude = np.expand_dims(x_grav[:,:,0], axis=2)
    for i in range(1,x_gyro.shape[2]):
        # angleAcc = math.degrees(math.atan(-acc[i,0]/math.sqrt(acc[i,1]**2+acc[i,2]**2)))
        new_attitude = (attitude[:,:,i-1] + x_gyro[:,:,i])*alpha + x_grav[:,:,i]*(1-alpha)
        new_attitude = np.expand_dims(new_attitude, axis=2)
        attitude     = np.concatenate((attitude, new_attitude), axis=2)
    
    X = np.concatenate((attitude, x_acc), axis=1)
    X = np.expand_dims(X, axis=1)
    
    return X