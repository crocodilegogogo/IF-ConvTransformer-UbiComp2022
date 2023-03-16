"""Collection of utility functions"""
from datetime import datetime
from logging import basicConfig, getLogger, Formatter, FileHandler, StreamHandler, DEBUG, WARNING
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from scipy.fftpack import fft
from utils.constants import INFERENCE_DEVICE

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
# from tensorflow import keras
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
import torch.nn.functional as F

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


def color_generator(i: int) -> str:
    l = ["#FFAF6D", "#DC4195", "#F1E898", "#6DCBB9", "#3E89C4", "#6F68CF"]
    return l[i]


def round_float(f: float, r: float = 0.000001) -> float:
    return float(Decimal(str(f)).quantize(Decimal(str(r)), rounding=ROUND_HALF_UP))


def round_list(l: List[float], r: float = 0.000001) -> List[float]:
    return [round_float(f, r) for f in l]


def round_dict(d: Dict[Any, Any], r: float = 0.000001) -> Dict[Any, Any]:
    return {key: round(d[key], r) for key in d.keys()}


def round(arg: Any, r: float = 0.000001) -> Any:
    if type(arg) == float or type(arg) == np.float64 or type(arg) == np.float32:
        return round_float(arg, r)
    elif type(arg) == list or type(arg) == np.ndarray:
        return round_list(arg, r)
    elif type(arg) == dict:
        return round_dict(arg, r)
    else:
        logger.error(f"Arg type {type(arg)} is not supported")
        return arg

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

def plot_feature_importance(
    models: List[Any],
    num_features: int,
    cols: List[str],
    importance_type: str = "gain",
    path: str = "importance.png",
    figsize: Tuple[int, int] = (16, 10),
    max_display: int = -1,
) -> None:
    """
    Args:
        importance_type: chosen from "gain" or "split"
    """
    importances = np.zeros((len(models), num_features))
    for i, model in enumerate(models):
        importances[i] = model.feature_importance(importance_type=importance_type)

    importance = np.mean(importances, axis=0)
    importance_df = pd.DataFrame({"Feature": cols, "Value": importance})
    importance_df = importance_df.sort_values(by="Value", ascending=False)[:max_display]

    plt.figure(figsize=figsize)
    sns.barplot(x="Value", y="Feature", data=importance_df)
    plt.title("Feature Importance (avg over folds)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def fft_transform(x, X_data, test_split):
    
    x_fft = fft(x)
    # X_data.shape[1]:9, the channel num of all sensor data.||x.shape[2]:8, the lenth of each time_step
    x_fft = x_fft.reshape([-1, X_data.shape[1], x.shape[1]])
    # 变一下维度，这样后面reshape时，每个sensor的xyz在一起
    if len(x_fft.shape) > 3:
        x_fft = np.swapaxes(x_fft.squeeze(),1,2)
    else:
        x_fft = np.swapaxes(x_fft,1,2)
    # split the data of different sensors
    x_sensor_fft = np.split(x_fft, int(x_fft.shape[2]/3), axis=2)
    x_all_sensor_fft = []
    # flatten the data of each sensor data, then concat them on the rows
    for i in range(len(x_sensor_fft)):
        shape_flag = x_sensor_fft[i].shape
        if i==0:
            fft_flag = x_sensor_fft[i].reshape([shape_flag[0], shape_flag[1]*shape_flag[2]])
            x_all_sensor_fft = fft_flag
        else:
            fft_flag = x_sensor_fft[i].reshape([shape_flag[0], shape_flag[1]*shape_flag[2]])
            x_all_sensor_fft = np.concatenate((x_all_sensor_fft, fft_flag), axis=1)
    # flatten the batch of real and imag values, get ready to merge them
    x_all_sensor_fft_real = x_all_sensor_fft.real.reshape(x_all_sensor_fft.shape[0]*x_all_sensor_fft.shape[1],1)
    x_all_sensor_fft_imag = x_all_sensor_fft.imag.reshape(x_all_sensor_fft.shape[0]*x_all_sensor_fft.shape[1],1)
    # concat the real and imag values on the columns, then reshape to the batch size to merge the real and imag values:
    x_merge_real_imag = np.concatenate((x_all_sensor_fft_real, x_all_sensor_fft_imag), axis=1)
    # a real value following a imag value
    x_merge_real_imag = x_merge_real_imag.reshape([x_all_sensor_fft.shape[0],-1])
    
    return x_merge_real_imag

def STFT_transform(X_data, y_data, STFT_intervals, test_split):
    
    STFT_result = []
    
    X_data = X_data.squeeze()
    torch_dataset = Data.TensorDataset(torch.FloatTensor(X_data), torch.tensor(y_data).long())
    data_loader = Data.DataLoader(dataset = torch_dataset,
                                  batch_size = X_data.shape[0] // test_split,
                                  shuffle = False)
    for step, (x,y) in enumerate(data_loader):
        
        x_split_merge_real_imag = []
        
        with torch.no_grad():
            x = x.cpu().data.numpy()
            if len(x.shape) == 2:
                x = np.expand_dims(x, axis=0)
            x = x.reshape([x.shape[0]*x.shape[1], x.shape[2]])
            if x.shape[1]%STFT_intervals == 0:
                x_split = np.split(x, STFT_intervals, axis=1)
            else:
                print('Please input a STFT_intervals value can be divided evently by data lenth:'+str(x.shape[1]))
            # Do fft transform for every time_step, then concat all the time_steps
            for time_step in range(len(x_split)):
                
                if time_step == 0:
                    x_split_merge_real_imag = fft_transform(x_split[time_step], X_data, test_split)
                    x_split_merge_real_imag = np.expand_dims(x_split_merge_real_imag, axis=1)
                else:
                    merge_cur_time_step = fft_transform(x_split[time_step], X_data, test_split)
                    merge_cur_time_step = np.expand_dims(merge_cur_time_step, axis=1)
                    x_split_merge_real_imag = np.concatenate((x_split_merge_real_imag, merge_cur_time_step), axis=1)
    
        if step == 0:
            STFT_result = x_split_merge_real_imag
        else:
            STFT_result = np.concatenate((STFT_result, x_split_merge_real_imag), axis=0)
    
    return np.expand_dims(STFT_result, axis=1)

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

def get_loso_train_test_data(SUBJECTS_IDS, All_users, All_data, All_labels):
    for (r_id, sub_id) in enumerate(SUBJECTS_IDS):
        if r_id == 0:
            ids = np.where(All_users == sub_id)[0]
        else:
            ids = np.concatenate((ids, np.where(All_users == sub_id)[0]))
    X = All_data[ids,:,:,:]
    y = All_labels[ids]
    return X, y

def logging_settings(classifier_name, CUR_DIR, dataset_name):
    
    # Logging settings
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

# !!!the below annotated code save logs of different classifiers to only one log file
# def logging_settings(classifier_name, CUR_DIR, dataset_name):
    
#     # Logging settings
#     EXEC_TIME = classifier_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
#     LOG_DIR = os.path.join(CUR_DIR, f"logs", dataset_name, classifier_name, f"{EXEC_TIME}")
#     MODEL_DIR = os.path.join(CUR_DIR, f"saved_model", dataset_name, classifier_name)
#     create_directory(LOG_DIR) # Create log directory
    
#     formatter = "%(levelname)s: %(asctime)s: %(filename)s: %(funcName)s: %(message)s"
#     basicConfig(filename=f"{LOG_DIR}/{EXEC_TIME}.log", level=DEBUG, format=formatter, filemode='w')
#     mpl_logger = getLogger("matplotlib")  # Suppress matplotlib logging
#     mpl_logger.setLevel(WARNING)
    
#     # Handle logging to both logging and stdout.
#     getLogger().addHandler(StreamHandler(sys.stdout))
    
#     logger = getLogger(__name__)
#     logger.setLevel(DEBUG)
#     logger.debug(f"{LOG_DIR}/{EXEC_TIME}.log")
    
#     return EXEC_TIME, LOG_DIR, MODEL_DIR, logger

def initialize_saving_variables(X_train, X_test, nb_classes, SUBJECT_NUM):
    
    # for test sets there are predictions for SUBJECT_NUM times
    # test_preds = np.zeros((SUBJECT_NUM, X_test.shape[0], nb_classes))
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

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Network_Total_Parameters:', total_num, 'Network_Trainable_Parameters:', trainable_num)
    return {'Total': total_num, 'Trainable': trainable_num}

def predict_tr_val_test(network, nb_classes, LABELS,
                        train_x, val_x, test_x,
                        train_y, val_y, test_y,
                        scores, per_training_duration,
                        run_id, output_directory_models,
                        test_split):
    
    start = time.time()
    
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
    
    end = time.time()
    
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
    
    return pred_train, pred_valid, pred_test, scores, (end-start)

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

def get_test_loss_acc_dynamic(net, loss_function, x_data, y_data, test_split=1, test_flag=False):
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
            output_bc = net(x, test_flag)[0]
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
            ## y = y.cuda() # 这个没用
            output_bc = net(x)[0]
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
            out = output_bc.cpu().data.numpy()
            # pred_bc = torch.max(output_bc, 1)[1].data.cuda().squeeze()               
            # pre = pred_bc.cpu().data.numpy()
            # if pre.size == 1:
            #     pre = pre.tolist()
            #     predict.append(pre)
            # else:
            #     pre = pre.tolist()
            #     predict.extend(pre)  
            output.extend(out)
    # acc = sum(predict == y_data)/y_data.shape[0]    
    return output

def save_models(net, output_directory_models, 
                loss_train, loss_train_results, 
                accuracy_validation, accuracy_validation_results, 
                start_time, training_duration_logs):   
    
    output_directory_best_val = output_directory_models+'best_validation_model.pkl'         
    if accuracy_validation >= max(accuracy_validation_results):        
        torch.save(net.state_dict(), output_directory_best_val)
    
    # # log training time 
    # training_duration = time.time() - start_time
    # training_duration_logs.append(training_duration)
    
    # return(training_duration_logs)

# def log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
#                 loss_validation_results, accuracy_validation_results, output_directory_models):
    
#     history = pd.DataFrame(data = np.zeros((EPOCH,5),dtype=np.float), 
#                            columns=['train_acc','train_loss','val_acc','val_loss','lr'])
#     history['train_acc'] = accuracy_train_results
#     history['train_loss'] = loss_train_results
#     history['val_acc'] = accuracy_validation_results
#     history['val_loss'] = loss_validation_results
#     history['lr'] = lr_results
    
#     # load saved models, predict, cal metrics and save logs    
#     history.to_csv(output_directory_models+'history.csv', index=False)
    
#     return history

# def plot_learning_history(EPOCH, history, path):
#     """Plot learning curve
#     Args:
#         fit (Any): History object
#         path (str, default="history.png")
#     """
#     fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
#     axL.plot(history["train_loss"], label="train")
#     axL.plot(history["val_loss"], label="validation")
#     axL.set_title("Loss")
#     axL.set_xlabel("epoch")
#     axL.set_ylabel("loss")
#     axL.legend(loc="upper right")

#     axR.plot(history["train_acc"], label="train")
#     axR.plot(history["val_acc"], label="validation")
#     axR.set_title("Accuracy")
#     axR.set_xlabel("epoch")
#     axR.set_ylabel("accuracy")
#     axR.legend(loc="upper right")

#     fig.savefig(path+'history.png')
#     plt.close()
    
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

def log_HyperParameters(logger, BATCH_SIZE, EPOCH, LR, subject_num):
    logger.debug("---HyperParameters---")
    logger.debug(f"BATCH_SIZE : {BATCH_SIZE}, EPOCH : {EPOCH}, LR : {LR}, SUBJECT_NUM : {subject_num}")

def log_redivdataset_network_info(logger, subject_id, X_tr, X_val, X_test, Y_tr, Y_val,
                                  y_test, nb_classes, classifier_parameter, classifier):
    if subject_id == 1:
        logger.debug("---Redivided dataset and network information---")
        logger.debug(f"X_train_shape={X_tr.shape}, X_validation_shape={X_val.shape}, X_test_shape={X_test.shape}")
        logger.debug(f"Y_train_shape={Y_tr.shape}, Y_validation_shape={Y_val.shape}, y_test_shape={y_test.shape}")
        logger.debug(f"num of categories = {nb_classes}")
        logger.debug(f"num of network parameter = {classifier_parameter}")
        logger.debug(f"the architecture of the network = {classifier}")

def log_every_SUBJECT_score(logger, log_training_duration, scores, label2act, nb_classes, SUBJECT_NUM):
    for i in range(SUBJECT_NUM):
        # Log Every Subject Scores
        logger.debug("---Per Subject Scores, Subject"+str(i)+"---")
        
        # log per SUBJECT training time
        logger.debug(f"Training Duration = {log_training_duration[i]}s")
        
        for mode in ["train", "valid", "test"]:
            # log the average of "logloss", "accuracy", "precision", "recall", "f1"
            logger.debug(f"---{mode}---")
            logger.debug(f"logloss={round(scores['logloss'][mode][i])}, accuracy={round(scores['accuracy'][mode][i])}, macro-precision={round(scores['macro-precision'][mode][i])}, macro-recall={round(scores['macro-recall'][mode][i])}, macro-f1={round(scores['macro-f1'][mode][i])}, weighted-f1={round(scores['weighted-f1'][mode][i])}, micro-f1={round(scores['micro-f1'][mode][i])}")
            # for metric in ["logloss", "accuracy", "macro-precision", "macro-recall", "macro-f1", "weighted-f1", "micro-f1"]:
            #     logger.debug(f"{metric}={round(np.mean(scores[metric][mode]))}")
            
            # # log the average of "per_class_f1"
            # class_f1_mat = scores["per_class_f1"][mode]
            # class_f1_result = {}
            # for class_id in range(nb_classes):
            #     per_class_f1 = class_f1_mat[i][class_id]
            #     class_f1_result[label2act[class_id]] = per_class_f1
            # logger.debug(f"per-class f1={round(class_f1_result)}")

def log_averaged_SUBJECT_scores(logger, log_training_duration, scores, label2act, nb_classes, SUBJECT_NUM):
    # Log Averaged Score of all Subjects
    logger.debug("---Subject Averaged Scores---")    
    # log the average of training time
    logger.debug(f"Averaged Training Duration = {(np.mean(log_training_duration))}s")
    
    for mode in ["train", "valid", "test"]:
        
        # log the average of "logloss", "accuracy", "precision", "recall", "f1"
        logger.debug(f"---{mode}---")
        for metric in ["logloss", "accuracy", "macro-precision", "macro-recall", "macro-f1", "weighted-f1", "micro-f1"]:
            logger.debug(f"{metric}={round(np.mean(scores[metric][mode]))} +- {round(np.std(scores[metric][mode]))}")
        
        # # log the average of "per_class_f1"
        # class_f1_mat = scores["per_class_f1"][mode]
        # class_f1_result = {}
        # for class_id in range(nb_classes):
        #     mean_class_f1 = np.mean([class_f1_mat[i][class_id] for i in range(SUBJECT_NUM)])
        #     class_f1_result[label2act[class_id]] = mean_class_f1
        # logger.debug(f"per-class f1={round(class_f1_result)}")

def log_inference_time(start1, end1, y_train, y_test, time2,
                       dataset_name, classifier_name, SUBJECT_NUM, logger):
    time1 = (end1 - start1)/(len(y_train)+len(y_test))
    time2 = time2/((len(y_train)+len(y_test))*SUBJECT_NUM)
    # time2 = time2/SUBJECT_NUM
    logger.debug("---Final Inference Time Per Sample-Averaged over Folds---")
    logger.debug(f"preprocessing_time={time1}")
    logger.debug(f"pure_inference_time={time2}")
    logger.debug(f"single_model_prepro_inference_time={time1+time2}")
    logger.debug(f"all_model_prepro_inference_time={time1+time2*SUBJECT_NUM}")
    # dataset_classifier_name = dataset_name + '_' + classifier_name
    # dataset_classifier_name_log.append(dataset_classifier_name)
    # time_log.append(time1+time2)
    # time1_log.append(time1)
    # time2_log.append(time2)
    # time_logs = dict(zip(dataset_classifier_name_log, time_log))
    
    return time1, time2

def plot_confusion_matrix(
    cms: Dict[str, np.ndarray],
    labels: Optional[List[str]] = None,
    path: str = "confusion_matrix.png",
) -> None:
    """Plot confusion matrix"""
    # Cal the ensembled confusion_matrix by averaging them
    cms = [np.mean(cms[mode], axis=0) for mode in ["train", "valid", "test"]]

    fig, ax = plt.subplots(ncols=3, figsize=(20, 7))
    for i, (cm, mode) in enumerate(zip(cms, ["train", "valid", "test"])):
        sns.heatmap(
            cm,
            annot=True,
            cmap="Blues",
            square=True,
            vmin=0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax[i],
        )
        ax[i].set_xlabel("Predicted label")
        ax[i].set_ylabel("True label")
        ax[i].set_title(f"Averaged confusion matrix - {mode}")

    plt.tight_layout()
    fig.savefig(path)
    plt.close()
    
def save_classifiers_comparison(MODELS_COMP_LOG_DIR, CLASSIFIERS, classifier_name,
                                scores, ALL_SUBJECTS_ID, preprocess_time, inference_time,
                                SUBJECT_NUM):
    
    for i in range(len(CLASSIFIERS)):
        if i == 0:
            CLASSIFIERS_names = CLASSIFIERS[0]+'&'
        else:
            CLASSIFIERS_names = CLASSIFIERS_names+CLASSIFIERS[i]+'&'
    classifiers_comparison_log_dir = MODELS_COMP_LOG_DIR + CLASSIFIERS_names + '-comparison' + '.csv'
    
    # record Averaged_SUBJECT_scores
    averaged_score_pd = pd.DataFrame(data = np.zeros((7, 1),dtype=np.str_),
                                     index=["accuracy","macro-precision",
                                            "macro-recall","macro-f1",
                                            "weighted-f1", "micro-f1", "inference-time"], 
                                     columns=[classifier_name])
    for row in averaged_score_pd.index:
        for column in averaged_score_pd.columns:
            if row != "inference-time":
                averaged_score_pd.loc[row][column] = np.str_(np.mean(scores[row]["test"])) + '+-' + np.str_(np.std(scores[row]["test"]))
            else:
                averaged_score_pd.loc[row][column] = np.str_(preprocess_time + inference_time)
    
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
        saved_averaged_scores  = pd.read_csv(classifiers_comparison_log_dir, skiprows=1, nrows=7, header=0, index_col=0)
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