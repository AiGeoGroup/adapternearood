import os
import sys
import shutil
from tqdm import tqdm

import numpy as np

from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
import torch
import torch.nn.functional as F

import seaborn as sns
from matplotlib import pyplot as plt

from adapter.libs.datautils import *
from adapter.libs.adapterutils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 4, 'pin_memory': True}

def get_folder_dataset(train_src_dir='/kaggle/input/nwpuresisc45/Dataset/train/train/', ood_number=5):
    # Load the dataset
    dataset = datasets.ImageFolder(train_src_dir)

    # Get LULC categories
    class_names = dataset.classes
    print("Class names: {}".format(class_names))
    print("Total number of classes: {}".format(len(class_names)))
    print(dataset.class_to_idx)

    # print(class_names[:ood_number])
    # print(class_names[ood_number:2*ood_number])
    return dataset

# 创建nwpuresisc OOD检测数据集,将原始数据拷贝到id_dst_dir
def create_nwpuresisc_subset(_src_dir, _dst_dir, ood_class_names):
    # mydataset = get_folder_dataset(train_src_dir)
    # ood_class_names = mydataset.classes[-ood_number:]
    os.makedirs(_dst_dir, exist_ok=True)
    for split in ['train', 'test']:  # 根据data设置文件构建in_data_set
        for class_name in ood_class_names:
            origin_data_dir = os.path.join(_src_dir, split + '/' + split,class_name)
            # print(origin_data_dir)
            in_dataset_dir = os.path.join(_dst_dir, split, class_name)
            # print(in_dataset_dir)
            shutil.copytree(origin_data_dir,
                            in_dataset_dir,
                            dirs_exist_ok=True)
    print("nwpuresisc_ood_subset")

# 创建OOD数据子集,将原始数据拷贝到id_dst_dir
def create_sallited_subset(_src_dir, _dst_dir, ood_class_names):
    # mydataset = get_folder_dataset(train_src_dir)
    # ood_class_names = mydataset.classes[-ood_number:]
    os.makedirs(_dst_dir, exist_ok=True)
    for split in ['train', 'test']:  # 根据data设置文件构建in_data_set
        for class_name in ood_class_names:
            origin_data_dir = os.path.join(_src_dir, split, class_name)
            # print(origin_data_dir)
            in_dataset_dir = os.path.join(_dst_dir, split, class_name)
            # print(in_dataset_dir)
            shutil.copytree(origin_data_dir,
                            in_dataset_dir,
                            dirs_exist_ok=True)
    print("Sallited_ood_subset")

# 获取ood数据集与loader
def get_ood_data_loader(
        nw_src_dir='/kaggle/input/rgbeurosat/RBG/',  # path to source data
        ood_dst_dir='/kaggle/working/OpenoodData/eurosatOOD/',  # path to out-of-distribution data set
        ood_number=5,
        sat_name='NWPU-RESISC45', 
        preprocess=None):
    
    test_src_dir = os.path.join(nw_src_dir, 'test')
    mydataset = get_folder_dataset(test_src_dir, ood_number)
    ood_class_names = mydataset.classes[ood_number:2 * ood_number]

    # 创建OOD检测数据集,将原始数据拷贝到id_dst_dir和ood_dst_dir目录下
    if not os.path.exists(ood_dst_dir):
        if sat_name == 'NWPU-RESISC45':
            create_nwpuresisc_subset(nw_src_dir, ood_dst_dir, ood_class_names)
        else:
            create_sallited_subset(nw_src_dir, ood_dst_dir, ood_class_names)

    test_dst_dir = os.path.join(ood_dst_dir, 'test')
    oodtestset = torchvision.datasets.ImageFolder(test_dst_dir,
                                                  transform=preprocess)

    oodtestloader = torch.utils.data.DataLoader(
        oodtestset,
        batch_size=configs['batch_size'],
        shuffle=False,
        num_workers=4)

    return ood_class_names, oodtestset, oodtestloader

# 获取id数据集与loader
def get_id_data_loader(
        nw_src_dir='/kaggle/input/rgbeurosat/RBG/',  # path to source data
        id_dst_dir='/kaggle/working/OpenoodData/eurosatID/',  # path to in-distribution data set
        ood_number=5,
        sat_name='NWPU-RESISC45', 
        preprocess=None):
    
    test_src_dir = os.path.join(nw_src_dir, 'test')
    mydataset = get_folder_dataset(test_src_dir, ood_number)
    id_class_names = mydataset.classes[:ood_number]

    # 创建OOD检测数据集,将原始数据拷贝到id_dst_dir和ood_dst_dir目录下
    if not os.path.exists(id_dst_dir):
        if sat_name == 'NWPU-RESISC45':
            create_nwpuresisc_subset(nw_src_dir, id_dst_dir, id_class_names)
        else:
            create_sallited_subset(nw_src_dir, id_dst_dir, id_class_names)

    test_dst_dir = os.path.join(id_dst_dir, 'test')
    idtestset = torchvision.datasets.ImageFolder(test_dst_dir,
                                                 transform=preprocess)

    idtestloader = torch.utils.data.DataLoader(
        idtestset,
        batch_size=configs['batch_size'],
        shuffle=False,
        num_workers=4)

    return id_class_names, idtestset, idtestloader

# 获取id数据集与loader
def extract_ID_features(
        configs,
        nw_src_dir='/kaggle/input/rgbeurosat/RBG/',  # path to source data
        id_data_path='/kaggle/working/OpenoodData/eurosatID/',
        ood_number=5,
        sat_name='NWPU-RESISC45',
        preprocess=None,
        clipmodel=None):
            
    ood_number=int(ood_number)
    id_class_names, idtestset, idtestloader = get_id_data_loader(
        nw_src_dir, id_data_path, ood_number, sat_name)

    test_id_labels = idtestset.classes
    # print(test_id_labels)

    id_val_features, id_val_labels = extract_features_from_loader(
        configs, "val", clipmodel, idtestloader)  # "val" is split
    id_test_features, id_test_labels = extract_features_from_loader(
        configs, "test", clipmodel, idtestloader)  # "test" is split
    return id_val_features, id_val_labels, id_test_features, id_test_labels, idtestset, idtestloader


def extract_OOD_features(
        configs,
        nw_src_dir='/kaggle/input/rgbeurosat/RBG/',  # path to source data
        ood_data_path='/kaggle/working/OpenoodData/eurosatOOD/',
        ood_number=5,
        sat_name='NWPU-RESISC45',
        preprocess=None,
        clipmodel=None):
    ood_number=int(ood_number)
    ood_class_names, oodtestset, oodtestloader = get_ood_data_loader(
        nw_src_dir, ood_data_path, ood_number, sat_name)

    test_ood_labels = oodtestset.classes
    # print(test_ood_labels)

    ood_val_features, ood_val_labels = extract_features_from_loader(
        configs, "val", clipmodel, oodtestloader)  # "val" is split
    ood_test_features, ood_test_labels = extract_features_from_loader(
        configs, "test", clipmodel, oodtestloader)  # "test" is split
    return ood_val_features, ood_val_labels, ood_test_features, ood_test_labels, oodtestset, oodtestloader
