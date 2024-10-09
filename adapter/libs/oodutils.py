import os
import shutil
import sys

import numpy as np
import seaborn as sns
import sklearn.metrics as sk
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from scipy.stats import entropy
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from adapter.libs.datautils import *
from adapter.libs.adapterutils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 4, 'pin_memory': True}


def get_ood_scores_clip(
        a_logits,
        id_class_nums,
        loader,
        in_dist=False,
        score_type='MCM',
        temperature=10,
        nood_beta=0.25):  #'MCM', 'energy', 'max-logit', 'entropy', 'var', 'maha'
    '''
    used for scores based on img-caption product inner products: MIP, entropy, energy score.
    '''
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []

    with torch.no_grad():
        if score_type == 'max-logit':
            smax = to_np(a_logits)
        elif score_type == 'MSP':
            smax = to_np(F.softmax(a_logits, dim=1))
        else:
            smax = to_np(F.softmax(a_logits / temperature, dim=1))

        if score_type == 'energy':  # Energy-based Out-of-distribution Detection (https://arxiv.org/pdf/2010.03759.pdf)
            # Energy = - T * logsumexp(logit_k / T), by default T = 1
            _score.append(-to_np(
                (temperature * torch.logsumexp(a_logits / temperature, dim=1)))
                          )  # energy score is expected to be smaller for ID
        elif score_type == 'entropy':
            # raw_value = entropy(smax)
            # filtered = raw_value[raw_value > -1e-5]
            _score.append(entropy(smax, axis=1))
            # _score.append(filtered)
        elif score_type == 'var':
            _score.append(-np.var(smax, axis=1))
        elif score_type in ['MCM', 'max-logit', 'MSP']:
            _score.append(-np.max(smax, axis=1))
        elif score_type == 'OMCM':
            smax = np.max(
                smax[:, :id_class_nums],
                axis=1) - nood_beta * np.max(smax[:, id_class_nums:], axis=1)
            _score.append(-smax)
        elif score_type == 'OODMCM':
            smax = np.max(
                smax[:, id_class_nums:],
                axis=1) - nood_beta * np.max(smax[:, :id_class_nums], axis=1)
            _score.append(-smax)

        # _score.append(-np.max(smax, axis=1))  # MCM
    return concat(_score)[:len(loader.dataset)].copy()


'''
Envisioning Outlier Exposure by Large Language Models for Out-of-Distribution Detection
'''

def get_ood_scores_clip_EOE(
    a_logits,
    id_class_nums,
    loader,
    in_dist=False,
    score_type='RelativeMCM',
    score_ablation='RelativeMCM',
    temperature=10,
    relative_beta=0.25):  #'MCM', 'energy', 'max-logit', 'entropy', 'var', 'maha'
    '''
    used for scores based on img-caption product inner products: MIP, entropy, energy score.
    '''
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []

    with torch.no_grad():
        if score_ablation in ['MAX', 'max-logit', 'energy']:
            smax = a_logits
        elif score_ablation == 'MSP':
            smax = to_np(F.softmax(a_logits, dim=1))
        else:
            smax = to_np(F.softmax(a_logits / temperature, dim=1))

        # cal score
        if score_type == 'RelativeMCM':
            if score_ablation == 'RelativeMCM':
                smax = np.max(smax[:, :id_class_nums],
                              axis=1) - relative_beta * np.max(
                                  smax[:, id_class_nums:], axis=1)
                _score.append(-smax)
            elif score_ablation == 'OODRMCM':
                smax = np.max(smax[:, id_class_nums:],
                              axis=1) - relative_beta * np.max(
                                  smax[:, :id_class_nums], axis=1)
                _score.append(-smax)

            elif score_ablation == 'MAX':
                iid_values, iid_indices = torch.max(smax[:, :id_class_nums],
                                                    dim=1)
                ood_values, ood_indices = torch.max(smax[:, id_class_nums:],
                                                    dim=1)
                condition = ood_values > iid_values
                smax[:, :id_class_nums][condition] = 1 / id_class_nums
                smax = smax[:, :id_class_nums]
                smax = to_np(F.softmax(smax / temperature, dim=1))
                smax = np.max(smax[:, :id_class_nums], axis=1)
                _score.append(-smax)
            elif score_ablation == 'MSP':
                smax = np.max(smax[:, :id_class_nums], axis=1)
                _score.append(-smax)
            elif score_ablation == 'energy':
                _score.append(
                    -to_np((temperature * torch.logsumexp(
                        smax[:, :id_class_nums] / temperature, dim=1)) -
                           (temperature * torch.logsumexp(
                               smax[:, id_class_nums:] / temperature, dim=1))))
            elif score_ablation == 'max-logit':
                # smax = torch.tensor(smax)
                _score.append(-to_np(
                    torch.max(smax[:, :id_class_nums], 1)[0] -
                    torch.max(smax[:, id_class_nums:], 1)[0]))
            else:
                raise NotImplementedError
        elif score_type == 'MCM':
            _score.append(-np.max(smax, axis=1))
        elif score_type == 'energy':
            #Energy = - T * logsumexp(logit_k / T), by default T = 1 in https://arxiv.org/pdf/2010.03759.pdf
            _score.append(-to_np(
                (temperature * torch.logsumexp(smax / temperature, dim=1)))
                          )  #energy score is expected to be smaller for ID
        elif score_type == 'max-logit':
            _score.append(-to_np(torch.max(smax, 1)[0]))

    return concat(_score)[:len(loader.dataset)].copy()



def get_mean_prec(clipmodel, train_loader, feat_dim=512, n_classes=45, normalize=True, device='cuda'): # feat=512 ViT or feat=1024 Resnet
    '''
    used for Mahalanobis score. Calculate class-wise mean and inverse covariance matrix
    '''
    classwise_mean = torch.empty(n_classes, feat_dim).to(device)
    all_features = []
    # classwise_features = []
    from collections import defaultdict
    classwise_idx = defaultdict(list)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.cuda()
            features = clipmodel.encode_image(images).float()
            features /= features.norm(dim=-1, keepdim=True)
            
            for label in labels:
                classwise_idx[label.item()].append(idx)
            all_features.append(features.cpu()) #for vit
            
    all_features = torch.cat(all_features)
    for cls in range(n_classes):
        classwise_mean[cls] = torch.mean(all_features[classwise_idx[cls]].float(), dim = 0)
        if normalize: 
            classwise_mean[cls] /= classwise_mean[cls].norm(dim=-1, keepdim=True)
    cov = torch.cov(all_features.T.double()) 
    precision = torch.linalg.inv(cov).float()
    print(f'cond number: {torch.linalg.cond(precision)}')
    return classwise_mean, precision


def get_Mahalanobis_score(configs, clipmodel, test_loader, classwise_mean, precision, in_dist = True, n_classes=45):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    '''
    # net.eval()
    Mahalanobis_score_all = []
    total_len = len(test_loader.dataset)
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            if (batch_idx >= total_len // configs['batch_size']) and in_dist is False:
                break   
            images, labels = images.cuda(), labels.cuda()
            features = clipmodel.encode_image(images).float()
            features /= features.norm(dim=-1, keepdim=True)
        
            for i in range(n_classes):
                class_mean = classwise_mean[i]
                zero_f = features - class_mean
                Mahalanobis_dist = -0.5*torch.mm(torch.mm(zero_f.to(device), precision.to(device)), zero_f.t()).diag()
                if i == 0:
                    Mahalanobis_score = Mahalanobis_dist.view(-1,1)
                else:
                    Mahalanobis_score = torch.cat((Mahalanobis_score, Mahalanobis_dist.view(-1,1)), 1)      
            Mahalanobis_score, _ = torch.max(Mahalanobis_score, dim=1)
            Mahalanobis_score_all.extend(-Mahalanobis_score.cpu().numpy())
        
    return np.asarray(Mahalanobis_score_all, dtype=np.float32)


def print_measures(log,
                   auroc,
                   aupr,
                   fpr,
                   method_name='Ours',
                   recall_level=0.95):
    if log == None:
        print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level),
                                            100 * fpr))
        print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
        print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))
    else:
        log.debug('\t\t\t\t' + method_name)
        log.debug('  FPR{:d} AUROC AUPR'.format(int(100 * recall_level)))
        log.debug('& {:.2f} & {:.2f} & {:.2f}'.format(100 * fpr, 100 * auroc,
                                                      100 * aupr))



def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out



def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None
            and not (np.array_equal(classes, [0, 1]) or np.array_equal(
                classes, [-1, 1]) or np.array_equal(classes, [0])
                     or np.array_equal(classes, [-1])
                     or np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl],
                                         1], np.r_[fps[sl],
                                                   0], np.r_[tps[sl],
                                                             0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true))
                          )  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])



import sklearn.metrics as sk


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr



def get_and_print_results(log,
                          in_score,
                          out_score,
                          auroc_list,
                          aupr_list,
                          fpr_list,
                          score_type='MCM'):
    '''
    1) evaluate detection performance for a given OOD test set (loader)
    2) print results (FPR95, AUROC, AUPR)
    '''
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0])
    auprs.append(measures[1])
    fprs.append(measures[2])
    print(
        f'in score samples (random sampled): {in_score[:3]}, out score samples: {out_score[:3]}'
    )
    # print(f'in score samples (min): {in_score[-3:]}, out score samples: {out_score[-3:]}')
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(
        fpr)  # used to calculate the avg over multiple OOD test sets
    print_measures(log, auroc, aupr, fpr, score_type)


def plot_distribution(log_directory, id_scores, ood_scores, out_dataset):
    sns.set(style="white", palette="muted")
    palette = ['#A8BAE3', '#55AB83']
    sns.displot({"ID":-1 * id_scores, "OOD": -1 * ood_scores}, label="id", kind = "kde", palette=palette, fill = True, alpha = 0.8)
    plt.savefig(os.path.join(log_directory, f"{out_dataset}.png"), bbox_inches='tight')
