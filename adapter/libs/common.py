import torch
import random
import numpy as np
import logging

# common.py
def get_num_cls(cfgs):
    NUM_CLS_DICT = {
        'Eurosat': 10,
        'NWPU-RESISC45': 45,
        'UC-Merced': 21, 
        'siri-wuhu': 12, 
        'AID': 30
    }
    n_cls = NUM_CLS_DICT[cfgs['in_dataset']]
    return n_cls


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def softmax(x):
    # 计算指数值
    exp_values = np.exp(x )
    # 计算每个样本的Softmax概率
    probabilities = exp_values / np.sum(exp_values)
    return probabilities



def setup_log(log_directory):
    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(log_directory, "ood_eval_info.log"), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    log.debug(f"#########logging############")
    return log
