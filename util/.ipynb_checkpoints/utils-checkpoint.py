import os
import random
import pandas as pd
import albumentations as A
import numpy as np
from sklearn import random_projection
import torch
from scipy.ndimage import zoom
import torch
import torch.nn as nn
from lifelines.utils import concordance_index

def save_checkpoint(model, fold, epoch, args, best_result=0, indicator='acc'):
    state_dict = model.state_dict()
    best_result = round(best_result, 4)
    if indicator == "latest":
        save_dict = {
            'epoch': epoch,
            'state_dict': state_dict
            }
        filename = f'{args.logdir}model/{fold}_{indicator}_model.pt'
    else:
        if indicator == "dice":
            save_dict = {
                    'epoch': epoch,
                    'best_dice': best_result,
                    'state_dict': state_dict
                    }
        elif indicator == "acc":
            save_dict = {
                    'epoch': epoch,
                    'best_acc': best_result,
                    'state_dict': state_dict
                    }
      
        elif indicator == "auc":
            save_dict = {
                    'epoch': epoch,
                    'best_auc': best_result,
                    'state_dict': state_dict
                    }
        else:
            save_dict = {
                    'epoch': epoch,
                    'best_acc': best_result,
                    'state_dict': state_dict
                    }
        filename = f'{args.logdir}model/{fold}_{indicator}_model.pt'

    torch.save(save_dict, filename)
    # print('Saving checkpoint', filename)



def seed_torch(seed=45):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# img_transform_train = A.Compose([  
#     A.VerticalFlip(p=0.5),
#     ]
#     ,additional_targets={
#         'image1': 'image',
#         'image2': 'image',
#         'image3': 'image',
#         'image4': 'image',
#         'image5': 'image',
#         'image6': 'image'
#     }
# )




def zscore_normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    return (X - mean) / std

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def resample_to_size(npy_image, target_size, order=1):
    source_size = npy_image.shape
    scale = np.array(target_size) / source_size
    # zoom_factor = source_size / np.array(target_size)
    target_npy_image = zoom(npy_image, scale, order=order)
    return target_npy_image


def resample_mask_to_size(npy_mask, target_size, num_label, order=1):
    source_size = npy_mask.shape
    scale = np.array(target_size) / source_size
    # zoom_factor = source_size / np.array(target_size)
    target_npy_mask = np.zeros_like(npy_mask)
    target_npy_mask = zoom(target_npy_mask, scale, order=order)
    for i in range(1, num_label + 1):
        current_mask = npy_mask.copy()

        current_mask[current_mask != i] = 0
        current_mask[current_mask == i] = 1

        current_mask = zoom(current_mask, scale, order=order)
        current_mask = (current_mask > 0.5).astype(np.uint8)
        target_npy_mask[current_mask != 0] = i
    return target_npy_mask

def createOnehotLabel(label, num_classes):
    onehot_label = np.zeros((num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
    for i in range(num_classes):
        onehot_label[i, :, :, :] = (label == i).astype(np.float32)
    return onehot_label

# def random_crop(img, seg=None):
#     if img.shape[0] == 128:
#         x_range, y_range = 96, 96
#     elif img.shape[0] == 96:
#         x_range, y_range = 64, 64

#     x_start = np.random.randint(0, img.shape[0] - x_range)
#     y_start = np.random.randint(0, img.shape[1] - y_range)

#     cropped_img = img[x_start:x_start + x_range, y_start:y_start + y_range, :]
#     if seg is not None:
#         cropped_seg = seg[x_start:x_start + x_range, y_start:y_start + y_range, :]
#         return cropped_img, cropped_seg
#     else:
#         return cropped_img

def get_ensemble_test_result_simple(logdir, phase='test'):
    score = 0
    for fold in range(5):
        time = np.load(f'{logdir}f{fold}_{phase}_y.npy')
        event = np.load(f'{logdir}f{fold}_{phase}_e.npy')
        score += np.load(f'{logdir}f{fold}_{phase}_output.npy')
        path = np.load(f'{logdir}f{fold}_{phase}_path.npy')

    score /= 5
    np.save(f'{logdir}ensemble_{phase}_e.npy', event)
    np.save(f'{logdir}ensemble_{phase}_y.npy', time)
    np.save(f'{logdir}ensemble_{phase}_output.npy', score)
    np.save(f'{logdir}ensemble_{phase}_path.npy', path)
    ensem_c = c_index(-score, time, event)
    return ensem_c

def get_ensemble_test_result(logdir, phase='test'):
    score = 0
    for fold in range(5):
        time = np.load(f'{logdir}f{fold}_{phase}_y.npy')
        event = np.load(f'{logdir}f{fold}_{phase}_e.npy')
        score += np.load(f'{logdir}f{fold}_{phase}_output.npy')
        path = np.load(f'{logdir}f{fold}_{phase}_path.npy')

        rfs_time = np.load(f'{logdir}f{fold}_{phase}_rfs_y.npy')
        rfs_event = np.load(f'{logdir}f{fold}_{phase}_rfs_e.npy')

    score /= 5
    np.save(f'{logdir}ensemble_{phase}_e.npy', event)
    np.save(f'{logdir}ensemble_{phase}_y.npy', time)
    np.save(f'{logdir}ensemble_{phase}_rfs_e.npy', rfs_event)
    np.save(f'{logdir}ensemble_{phase}_rfs_y.npy', rfs_time)
    np.save(f'{logdir}ensemble_{phase}_output.npy', score)
    np.save(f'{logdir}ensemble_{phase}_path.npy', path)
    ensem_c = c_index(-score, time, event)
    return ensem_c
    
def save_ensemble_result_simple(path, phase):
    csv_path = f'{path}ensemble_{phase}.csv'
    print('saving_ensemble_result', os.path.exists(csv_path), csv_path)
    if not os.path.exists(csv_path):
        time = np.load(f'{path}ensemble_{phase}_y.npy')
        event = np.load(f'{path}ensemble_{phase}_e.npy')
        score = np.load(f'{path}ensemble_{phase}_output.npy')
        img_path = np.load(f'{path}ensemble_{phase}_path.npy')
        # img_path = np.expand_dims(img_path, 1)
   
        all_info = np.hstack((img_path, time, event, score))
        all_info = pd.DataFrame(all_info, columns=['path', 'time', 'event', 'score'])
        print('saving_ensemble_result')
        all_info.to_csv(csv_path, index=False)


def save_ensemble_result(path, phase):
    csv_path = f'{path}ensemble_{phase}.csv'
    print('saving_ensemble_result', os.path.exists(csv_path), csv_path)
    csv_pat_rfs = f'{path}ensemble_{phase}_rfs.csv'
    if not os.path.exists(csv_path):
        time = np.load(f'{path}ensemble_{phase}_y.npy')
        event = np.load(f'{path}ensemble_{phase}_e.npy')
        score = np.load(f'{path}ensemble_{phase}_output.npy')
        img_path = np.load(f'{path}ensemble_{phase}_path.npy')
        rfs_time = np.load(f'{path}ensemble_{phase}_rfs_y.npy')
        rfs_event = np.load(f'{path}ensemble_{phase}_rfs_e.npy')
        img_path = np.expand_dims(img_path, 1)

        all_info = np.hstack((img_path, time, event, score))
        all_info = pd.DataFrame(all_info, columns=['path', 'time', 'event', 'score'])
        print('saving_ensemble_result')
        all_info.to_csv(csv_path, index=False)

        rfs_all_info = np.hstack((img_path, rfs_time, rfs_event, score))
        rfs_all_info = pd.DataFrame(rfs_all_info, columns=['path', 'rfstime', 'rfsevent', 'score'])
        print('saving_ensemble_result rfs')
        rfs_all_info.to_csv(csv_pat_rfs, index=False)
        

def preprocess_img(ori_img, transform=None):

    img = ori_img.astype(np.float32)        

    # img = normalization(img)
    img = zscore_normalize(img)

    # if resample_size:
    img = resample_to_size(img, (5, 64, 64))
    # img = resample_to_size(img, (64, 64, 64))

    if transform:
        trans_img = apply_transform(transform, img.astype(np.float32))
        trans_img = torch.tensor(trans_img)
         
    else:
        trans_img = torch.tensor(img)
        trans_img = trans_img.unsqueeze(dim=0)
    return trans_img


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR

    Youden_index = np.argmax(y)  # Only the first occurrence is returned.

    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point



def c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)

def set_expname(args):
    if args.model_name in ("resnet_18", 'resnet_34', 'resnet_50'):
        args.pretrained_dir = f'{args.pretrained_dir}{args.model_name}.pth'
    
    
    if args.iinput == -1 and (args.cinput != -1 or args.sinput != -1):
        if args.sinput != -1 and args.cinput == -1:
            args.json_list = f'/mnt/ExtData/hcc/preprocess/json/block_{args.data_version}_seq.json'
            args.test_json_list = f'/mnt/ExtData/hcc/preprocess/json/block_{args.data_version}_test_seq.json'
        else:
            args.json_list = f'/mnt/ExtData/hcc/preprocess/json/block_{args.data_version}.json'
            args.test_json_list = f'/mnt/ExtData/hcc/preprocess/json/block_{args.data_version}_test.json'
    else:
        args.json_list = f'/mnt/ExtData/hcc/preprocess/json/block_{args.data_version}_img.json'
        args.test_json_list = f'/mnt/ExtData/hcc/preprocess/json/block_{args.data_version}_test_img.json'

    
    
    train_name = f'{args.label_type}_{args.label_name}'
    if args.multimod:
        train_name = f'{train_name}_multimod'
    else:
        train_name = f'{train_name}_singlemod{args.mod_idx}'

    return train_name


def get_binary_pred(p, threshold = 0.5):
    binary_predict = copy(p).detach().cpu()
    binary_predict[binary_predict > threshold] = 1
    binary_predict[binary_predict <= threshold] = 0
    return binary_predict

def MacroAUC(output, label):
    y_pred = output #num_instance*num_label
    y_true = label #num_instance*num_label 
    num_instance,num_class =   y_pred.shape
    count = np.zeros((num_class,1))   # store the number of postive instance'score>negative instance'score
    num_P_instance =  np.zeros((num_class,1)) #number of positive instance for every label      
    num_N_instance =  np.zeros((num_class,1)) 
    AUC = np.zeros((num_class,1))  # for each label
    count_valid_label = 0
    for  i in range(num_class): #第i类
        num_P_instance[i,0] = sum(y_true[:,i] == 1) #label,,test_target
        num_N_instance[i,0] = num_instance - num_P_instance[i,0]
        # exclude the label on which all instances are positive or negative,
        # leading to num_P_instance(i,1) or num_N_instance(i,1) is zero
        if num_P_instance[i,0] == 0 or num_N_instance[i,0] == 0:
            AUC[i,0] = 0
            count_valid_label = count_valid_label + 1
        else:

            temp_P_Outputs = np.zeros((int(num_P_instance[i,0]), num_class))
            temp_N_Outputs = np.zeros((int(num_N_instance[i,0]), num_class))
            #
            temp_P_Outputs[:,i] = y_pred[y_true[:,i]==1,i]
            temp_N_Outputs[:,i] = y_pred[y_true[:,i]==0,i]    
            for m in range(int(num_P_instance[i,0])):
                for n in range(int(num_N_instance[i,0])):
                    if(temp_P_Outputs[m,i] > temp_N_Outputs[n,i] ):
                        count[i,0] = count[i,0] + 1
                    elif(temp_P_Outputs[m,i] == temp_N_Outputs[n,i]):
                        count[i,0] = count[i,0] + 0.5
            
            AUC[i,0] = count[i,0]/(num_P_instance[i,0]*num_N_instance[i,0])  
    macroAUC1 = sum(AUC)/(num_class-count_valid_label)
    return  float(macroAUC1), AUC    
