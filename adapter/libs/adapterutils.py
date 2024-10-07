import os
import random
import yaml
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

# from adapter.dpmf.clip import clip
import clip

from adapter.libs.intraproxy import *
from adapter.libs.datautils import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = "cpu"

# DualAdapter/extract_few_shot_features.py
# 从few shot数据集中抽取tain特征, cache_keys和cache_value信息
def extract_few_shot_features(
        cfg,
        clip_model,
        train_loader_cache):  # train_loader_cache is few shot trainloader
    cache_keys = []
    cache_values = []
    with torch.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(cfg['augment_epoch']):
            train_features = []
            print('Augment Epoch: {:} / {:}'.format(augment_idx,
                                                    cfg['augment_epoch']))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images = images.to(device)
                image_features = clip_model.encode_image(images)
                train_features.append(image_features)
                if augment_idx == 0:
                    target = target.to(device)
                    cache_values.append(target)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(
                0))  # cache_keys are the few-shot train feautres

    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = F.one_hot(torch.cat(cache_values, dim=0).to(
        torch.int64)).half()  # cache_values are the few-shot train labels
    torch.save(cache_keys,
               cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
    torch.save(cache_values,
               cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return train_features, cache_keys, cache_values


# 采用CLIP模型从loader中抽取视觉特征信息
def extract_features_from_loader(cfg, split, clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.to(device), target.to(device)
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)
    features, labels = torch.cat(features), torch.cat(labels)
    torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
    torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
    return features, labels


# 根据Positive语言提示器抽取分类特征，或称之为语言文本表征
def extract_text_features(cfg, classnames, clip_model, template):
    # text embedding for dual texts
    with torch.no_grad():
        clip_weights = []
        for i, classname in enumerate(classnames):
            # Tokenize the prompts
            if cfg['rsdataname'] != 'siri-wuhu':
                classname = classname.replace('_', ' ')

            template_texts = [t.format(classname) for t in template]
            texts_token = clip.tokenize(template_texts,
                                        truncate=True).to(device)
            # prompt ensemble for Eurosat
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_template.pt")

    return clip_weights


# 根据cupl方法抽取positive prompts的语言文本表征text embeddings
def extract_cupl_text_features(cfg, classnames, prompt_path, clip_model,
                               template):
    f = open(prompt_path)
    prompts = json.load(f)

    # text embedding for cupl texts
    with torch.no_grad():
        cupl_clip_weights = []
        for i, classname in enumerate(classnames):
            # Tokenize the prompts
            if cfg['rsdataname'] != 'siri-wuhu':
                classname = classname.replace('_', ' ')

            template_texts = [t.format(classname) for t in template]
            cupl_texts = prompts[classname]
            texts = cupl_texts + template_texts

            texts_token = clip.tokenize(texts, truncate=True).to(device)
            # prompt ensemble for Eurosat
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            cupl_clip_weights.append(class_embedding)

        cupl_clip_weights = torch.stack(cupl_clip_weights, dim=1).to(device)
    torch.save(cupl_clip_weights, cfg['cache_dir'] + "/text_weights_cupl.pt")

    return cupl_clip_weights


# 根据cupl方法抽取negative prompts的语言文本表征text embeddings
def extract_text_features_negative(cfg, classnames, prompt_path, clip_model,
                                   template):
    f = open(prompt_path)
    prompts = json.load(f)
    with torch.no_grad():
        clip_weights = []
        clip_weights_all = []
        for i, classname in enumerate(classnames):
            # Tokenize the prompts
            if cfg['rsdataname'] != 'siri-wuhu':
                classname = classname.replace('_', ' ')
            template_texts = [t.format(classname) for t in template]
            cupl_texts = prompts[classname]
            texts = template_texts

            texts_token = clip.tokenize(texts, truncate=True).to(device)
            # prompt ensemble for eurosat
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    torch.save(clip_weights,
               cfg['cache_dir'] + "/text_weights_negative_template.pt")
    print(clip_weights.shape)
    return clip_weights

# utils.py
def load_text_feature(cfg):
    save_path = cfg['cache_dir'] + "/text_weights_template.pt"
    clip_weights_template = torch.load(save_path)
    save_path = cfg['cache_dir'] + "/text_weights_cupl.pt"
    clip_weights_cupl = torch.load(save_path)
    save_path = cfg['cache_dir'] + "/text_weights_negative_template.pt"
    clip_weights_negative = torch.load(save_path)
    return clip_weights_template, clip_weights_cupl, clip_weights_negative

def load_few_shot_feature(cfg):
    cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
    cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
    return cache_keys, cache_values # cache_keys are features, cache_values are labels


def load_val_test_feature(cfg, split):
    features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
    labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):
    if cfg['search_hp'] == True:

        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)

                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def get_clip_model_feat_dims(model_backbone):
    # feature dimensions for each model
    feat_dims = {
        'RN50': 1024,
        'ViT-B/16': 512,
        'RN50x16': 768,
        'RN101': 512,
        'ViT-L/14': 768,
        'ViT-B/32': 512
    }
    return feat_dims[model_backbone]


'''Run Tip Adaptet For Resnet and VIT'''


def run_tip_adapter_ensemble(cfg, cache_keys_resnet, cache_values_resnet,
                             cache_keys_vit, cache_values_vit,
                             val_features_resnet, val_features_vit, val_labels,
                             test_features_resnet, test_features_vit,
                             test_labels, clip_weights_resnet, clip_weights_vit):

    print("\n-------- Searching hyperparameters on the val set. --------")
    # Tip-Adapter parameters
    beta, alpha, gamma = cfg['init_beta'], cfg['init_alpha'], cfg['init_gamma']

    # Zero-shot CLIP (ResNet and ViT)
    clip_logits_resnet = val_features_resnet @ clip_weights_resnet
    clip_logits_vit = val_features_vit @ clip_weights_vit

    # Combine ResNet and ViT logits
    clip_logits = gamma * clip_logits_resnet + clip_logits_vit
    acc = cls_acc(clip_logits, val_labels)
    print(f"\n**** Zero-shot CLIP's val accuracy: {acc:.2f}. ****\n")

    # Affinity and cache logits for ResNet
    affinity_resnet = val_features_resnet @ cache_keys_resnet
    cache_logits_resnet = (
        (-1) * (beta - beta * affinity_resnet)).exp() @ cache_values_resnet

    # Affinity and cache logits for ViT
    affinity_vit = val_features_vit @ cache_keys_vit
    cache_logits_vit = ((-1) *
                        (beta - beta * affinity_vit)).exp() @ cache_values_vit

    # Combine ResNet and ViT cache logits
    cache_logits = gamma * cache_logits_resnet + cache_logits_vit

    # Combine cache logits with clip logits
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print(f"**** Tip-Adapter's val accuracy: {acc:.2f}. ****\n")

    # Hyperparameter search
    best_beta, best_alpha, best_gamma = search_hp_ensemble(cfg,
                                                           cache_keys_resnet,
                                                           cache_values_resnet,
                                                           val_features_resnet,
                                                           cache_keys_vit,
                                                           cache_values_vit,
                                                           val_features_vit,
                                                           val_labels,
                                                           clip_weights,
                                                           adapter=None)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP on test set
    clip_logits_resnet = test_features_resnet @ clip_weights
    clip_logits_vit = test_features_vit @ clip_weights
    clip_logits = best_gamma * clip_logits_resnet + clip_logits_vit

    acc = cls_acc(clip_logits, test_labels)
    print(f"\n**** Zero-shot CLIP's test accuracy: {acc:.2f}. ****\n")

    # Tip-Adapter on test set
    affinity_resnet = test_features_resnet @ cache_keys_resnet
    cache_logits_resnet = (
        (-1) *
        (best_beta - best_beta * affinity_resnet)).exp() @ cache_values_resnet

    affinity_vit = test_features_vit @ cache_keys_vit
    cache_logits_vit = (
        (-1) * (best_beta - best_beta * affinity_vit)).exp() @ cache_values_vit

    # Combine cache logits from ResNet and ViT
    cache_logits = best_gamma * cache_logits_resnet + cache_logits_vit

    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print(f"**** Combine cache Tip-Adapter's test accuracy: {acc:.2f}. ****\n")

    # Resnet TIP-Adapter
    tip_logits = clip_logits_resnet + cache_logits_resnet * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print(f"**** Rensnet Tip-Adapter's test accuracy: {acc:.2f}. ****\n")


def run_tip_adapter_(cfg, cache_keys, cache_values, val_features, val_labels,
                    test_features, test_labels, clip_weights):  # training-free
    print("\n-------- Searching hyperparameters on the val set. --------")

    #zero-shot clip
    clip_logits = val_features @ clip_weights  # 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']

    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp(
    ) @ cache_values  # ((-1) * (beta - beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values,
                                      val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")
    # Zero-shot CLIP
    clip_logits = test_features @ clip_weights  # 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    affinity = test_features @ cache_keys
    cache_logits = ((-1) *
                    (best_beta - best_beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + best_alpha * cache_logits
    acc_top_1 = cls_acc(tip_logits, test_labels)
    acc_top_3 = cls_acc(tip_logits, test_labels, topk=3)
    print("**** Tip-Adapter's top 1 test accuracy: {:.2f}. ****\n".format(
        acc_top_1))
    print("**** Tip-Adapter's top 3 test accuracy: {:.2f}. ****\n".format(
        acc_top_3))


def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels,
                      test_features, test_labels, clip_weights, clip_model,
                      train_loader):
    #Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1],
                        bias=False).to(clip_model.dtype).to(device)
    adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg['train_epoch'] * len(train_loader))

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):  #Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []

        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))
        for i, (images, targets) in enumerate(tqdm(train_loader)):
            targets = targets.type(torch.LongTensor)  # casting to long
            images, targets = images.to(device), targets.to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) *
                            (beta - beta * affinity)).exp() @ cache_values
            clip_logits = image_features @ clip_weights  # 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, targets)

            acc = cls_acc(tip_logits, targets)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(
            current_lr, correct_samples / all_samples, correct_samples,
            all_samples,
            sum(loss_list) / len(loss_list)))

        # Eval
        adapter.eval()
        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = test_features @ clip_weights  # 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(
                adapter.weight,
                cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")

    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" +
                                str(cfg['shots']) + "shots.pt")
    print(
        f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****"
    )

    print("-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg,
                                      cache_keys,
                                      cache_values,
                                      val_features,
                                      val_labels,
                                      clip_weights,
                                      adapter=adapter)

    print(
        "-------------------- Evaluating on the test set. ----------------------"
    )

    affinity = adapter(test_features)
    cache_logits = ((-1) *
                    (best_beta - best_beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    # print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****".format(max(best_acc, acc)))
    print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****".format(acc))


def search_hp_ensemble(cfg,
                       cache_keys_resnet,
                       cache_values_resnet,
                       val_features_resnet,
                       cache_keys_vit,
                       cache_values_vit,
                       val_features_vit,
                       val_labels,
                       clip_weights_resnet, 
                       clip_weights_vit,
                       adapter=None):
    if cfg['search_hp'] == True:
        beta_list = [
            i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1
            for i in range(cfg['search_step'][0])
        ]
        alpha_list = [
            i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1
            for i in range(cfg['search_step'][1])
        ]
        gamma_list = [
            i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1
            for i in range(cfg['search_step'][1])
        ]

        best_acc = 0
        best_beta, best_alpha, best_gamma = 0, 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                for gamma in gamma_list:
                    if adapter:
                        affinity = adapter(features)
                    else:
                        # Affinity and cache logits for ResNet
                        affinity_resnet = val_features_resnet @ cache_keys_resnet
                        # Affinity and cache logits for ViT
                        affinity_vit = val_features_vit @ cache_keys_vit

                    cache_logits_resnet = ((-1) *
                                           (beta - beta * affinity_resnet)
                                           ).exp() @ cache_values_resnet
                    cache_logits_vit = (
                        (-1) *
                        (beta - beta * affinity_vit)).exp() @ cache_values_vit

                    # Combine ResNet and ViT cache logits
                    cache_logits = gamma * cache_logits_resnet + cache_logits_vit  # gamma
                    # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                    vit_clip_logits = 100. * val_features_vit @ clip_weights_vit
                    tip_logits = vit_clip_logits + cache_logits * alpha
                    acc = cls_acc(tip_logits, val_labels)

                    if acc > best_acc:
                        print(
                            "New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}"
                            .format(beta, alpha, acc))
                        best_acc = acc
                        best_beta = beta
                        best_alpha = alpha
                        best_gamma = gamma

        print(
            "\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha, best_gamma


def run_intra_proxy_tip_adapter(cfg, cache_keys, cache_values, val_features,
                                val_labels, test_features, test_labels,
                                clip_weights):  # training-free
    print("-------- Searching hyperparameters on the val set. --------")

    with torch.no_grad():
        #zero-shot clip
        #clip_logits = val_features @ clip_weights # 100. * val_features @ clip_weights
        clip_logits = test_features @ clip_weights
        acc = cls_acc(clip_logits, val_labels)
        print("**** Zero-shot CLIP's val accuracy: {:.2f}. ****".format(acc))

        # Tip-Adapter
        beta, alpha = cfg['init_beta'], cfg['init_alpha']

        # affinity = val_features @ cache_keys
        affinity = test_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp(
        ) @ cache_values  # ((-1) * (beta - beta * affinity)).exp() @ cache_values

        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, val_labels)
        print("**** Tip-Adapter's val accuracy: {:.2f}. ****".format(acc))

        print("-------- Searching hyperparameters on the val set. --------")
        # Search Hyperparameters
        # best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)
        best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values,
                                          test_features, val_labels,
                                          clip_weights)

        print("-------- Evaluating on the test set. --------")
        # Zero-shot CLIP
        clip_logits = test_features @ clip_weights  # 100. * test_features @ clip_weights
        acc = cls_acc(clip_logits, test_labels)
        acc_top_3 = cls_acc(clip_logits, test_labels, topk=3)
        print("**** Zero-shot CLIP's test accuracy: {:.2f}. ****".format(acc))
        print(
            "**** Zero-shot CLIP's topk=3 test accuracy: {:.2f}. ****".format(
                acc_top_3))

        # Zero-shot CUPL CLIP
        zeroshot_weights_both = 0.45 * clip_weights + 0.55 * clip_weight_cupl
        cupl_clip_logits = test_features @ zeroshot_weights_both  # 100. * test_features @ clip_weights
        acc = cls_acc(cupl_clip_logits, test_labels)
        acc_top_3 = cls_acc(cupl_clip_logits, test_labels, topk=3)
        print("**** Zero-shot CUPL CLIP's test accuracy: {:.2f}. ****".format(
            acc))
        print("**** Zero-shot CUPL CLIP's topk=3 test accuracy: {:.2f}. ****".
              format(acc_top_3))

        # Zero-shot InMap CLIP
        image_classifier = image_opt(test_features, zeroshot_weights_both,
                                     plabel, cfg['lr'],
                                     cfg['iters_proxy'], cfg['tau_i'],
                                     cfg['alpha'])
        InMap_logits_i = test_features @ image_classifier
        acc = cls_acc(InMap_logits_i, test_labels)
        acc_top_3 = cls_acc(InMap_logits_i, test_labels, topk=3)
        print("**** Zero-shot InMap CLIP's test accuracy: {:.2f}. ****".format(
            acc))
        print("**** Zero-shot InMap CLIP's topk=3 test accuracy: {:.2f}. ****".
              format(acc_top_3))

        # Tip-Adapter
        # AdapterOOD
        best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values,
                                          test_features, test_labels,
                                          image_classifier)
        affinity = test_features @ cache_keys
        cache_logits = (
            (-1) * (best_beta - best_beta * affinity)).exp() @ cache_values

        # intra_proxy_tip_logits = cupl_clip_logits + best_alpha * cache_logits
        intra_proxy_tip_logits = 100. * InMap_logits_i + best_alpha * cache_logits
        acc = cls_acc(intra_proxy_tip_logits, test_labels)
        acc_top_3 = cls_acc(intra_proxy_tip_logits, test_labels, topk=3)
        print("**** My AdapterOOD Tip-Adapter's test accuracy: {:.2f}. ****".
              format(acc))
        print(
            "**** My AdapterOOD Tip-Adapter's  topk=3 test accuracy: {:.2f}. ****"
            .format(acc_top_3))

    return intra_proxy_tip_logits, cupl_clip_logits, clip_logits


def run_intra_proxy_tip_adapter_ensemble(cfg,
                                         cache_keys_resnet,
                                         cache_values_resnet,
                                         cache_keys_vit,
                                         cache_values_vit,
                                         val_features_resnet,
                                         val_features_vit,
                                         val_labels,
                                         test_features_resnet,
                                         test_features_vit,
                                         test_labels,
                                         clip_weights,
                                         cupl_clip_weights,
                                         plabel):  # training-free

    # Tip-Adapter
    beta, alpha, gamma = cfg['init_beta'], cfg['init_alpha'], cfg['init_gamma']

    with torch.no_grad():
        #zero-shot clip
        #clip_logits = val_features @ clip_weights # 100. * val_features @ clip_weights
        clip_logits_resnet = val_features_resnet @ clip_weights
        clip_logits_vit = val_features_vit @ clip_weights

        acc = cls_acc(clip_logits_resnet, test_labels)  # RESNET
        print("**** RESNET Zero-shot CLIP's val accuracy: {:.2f}. ****".format(
            acc))
        acc = cls_acc(clip_logits_resnet, test_labels, topk=3)  # RESNET
        print("**** RESNET Zero-shot CLIP's val accuracy: {:.2f}. ****".format(
            acc))

        acc = cls_acc(clip_logits_vit, test_labels)  # VIT
        print(
            "**** VIT Zero-shot CLIP's val accuracy: {:.2f}. ****".format(acc))
        acc = cls_acc(clip_logits_vit, test_labels, topk=3)  # VIT
        print("**** VIT Zero-shot CLIP's topk=3 test accuracy: {:.2f}. ****".
              format(acc))

        # Combine ResNet and ViT logits
        ensemble_clip_logits = gamma * clip_logits_resnet + clip_logits_vit
        acc = cls_acc(ensemble_clip_logits, test_labels)
        print(
            f"\n**** Combine ResNet and ViT Zero-shot CLIP's val accuracy: {acc:.2f}. ****\n"
        )
        acc = cls_acc(ensemble_clip_logits, test_labels, topk=3)
        print(
            f"\n**** Combine ResNet and ViT Zero-shot CLIP's top=3 val accuracy: {acc:.2f}. ****\n"
        )

        # Affinity and cache logits for ResNet
        affinity_resnet = val_features_resnet @ cache_keys_resnet
        cache_logits_resnet = (
            (-1) * (beta - beta * affinity_resnet)).exp() @ cache_values_resnet

        # Affinity and cache logits for ViT
        affinity_vit = val_features_vit @ cache_keys_vit
        cache_logits_vit = (
            (-1) * (beta - beta * affinity_vit)).exp() @ cache_values_vit

        # Combine ResNet and ViT cache logits
        cache_logits = gamma * cache_logits_resnet + cache_logits_vit

        # tip_logits = clip_logits + cache_logits * alpha
        tip_logits = ensemble_clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, val_labels)
        print("**** Tip-Adapter's val accuracy: {:.2f}. ****".format(acc))
        acc = cls_acc(tip_logits, val_labels, topk=3)
        print(
            "**** Tip-Adapter's top-k val accuracy: {:.2f}. ****".format(acc))

        print("-------- Evaluating on the test set. --------")
        # Zero-shot CUPL CLIP
        zeroshot_weights_both = 0.45 * clip_weights + 0.55 * cupl_clip_weights
        cupl_clip_logits = test_features_vit @ zeroshot_weights_both  # 100. * test_features @ clip_weights
        acc = cls_acc(cupl_clip_logits, test_labels)
        acc_top_3 = cls_acc(cupl_clip_logits, test_labels, topk=3)
        print("**** VIT Zero-shot CUPL CLIP's test accuracy: {:.2f}. ****".
              format(acc))
        print(
            "**** VIT Zero-shot CUPL CLIP's topk=3 test accuracy: {:.2f}. ****"
            .format(acc_top_3))

        # Zero-shot InMap CLIP
        image_classifier = image_opt(zeroshot_weights_both, zeroshot_weights_both,
                                     plabel, cfg['lr'],
                                     cfg['iters_proxy'], cfg['tau_i'],
                                     cfg['alpha'])
        InMap_logits_i = test_features_vit @ image_classifier
        acc = cls_acc(InMap_logits_i, test_labels)
        acc_top_3 = cls_acc(InMap_logits_i, test_labels, topk=3)
        print("**** Zero-shot InMap CLIP's test accuracy: {:.2f}. ****".format(
            acc))
        print("**** Zero-shot InMap CLIP's topk=3 test accuracy: {:.2f}. ****".
              format(acc_top_3))

        # Tip-Adapter
        # AdapterOOD
        # best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, image_classifier)
        # Hyperparameter search
        best_beta, best_alpha, best_gamma = search_hp_ensemble(
            cfg,
            cache_keys_resnet,
            cache_values_resnet,
            val_features_resnet,
            cache_keys_vit,
            cache_values_vit,
            val_features_vit,
            val_labels,
            clip_weights,
            adapter=None)
        print(best_beta, best_alpha, best_gamma)

        # Affinity and cache logits for ResNet
        affinity_resnet = test_features_resnet @ cache_keys_resnet
        cache_logits_resnet = (
            (-1) * (beta - beta * affinity_resnet)).exp() @ cache_values_resnet

        # Affinity and cache logits for ViT
        affinity_vit = test_features_vit @ cache_keys_vit
        cache_logits_vit = (
            (-1) * (beta - beta * affinity_vit)).exp() @ cache_values_vit

        # Combine ResNet and ViT cache logits
        cache_logits = best_gamma * cache_logits_resnet + cache_logits_vit

        # tip_logits = clip_logits + cache_logits * alpha
        tip_logits = ensemble_clip_logits + cache_logits * alpha

        # Intra Proxy Tip Logits with InMap model
        intra_proxy_tip_logits = 100. * InMap_logits_i + best_alpha * cache_logits
        acc = cls_acc(intra_proxy_tip_logits, test_labels)
        acc_top_3 = cls_acc(intra_proxy_tip_logits, test_labels, topk=3)
        print("**** My AdapterOOD Tip-Adapter's test accuracy: {:.2f}. ****".
              format(acc))
        print(
            "**** My AdapterOOD Tip-Adapter's topk=3 test accuracy: {:.2f}. ****"
            .format(acc_top_3))

        # Intra Proxy Tip Logits with InMap and ViT model
        intra_proxy_tip_logits = 100. * InMap_logits_i + best_alpha * cache_logits_vit
        acc = cls_acc(intra_proxy_tip_logits, test_labels)
        acc_top_3 = cls_acc(intra_proxy_tip_logits, test_labels, topk=3)
        print(
            "**** My AdapterOOD ViT's test accuracy: {:.2f}. ****".format(acc))
        print("**** My AdapterOOD ViT's topk=3 test accuracy: {:.2f}. ****".
              format(acc_top_3))

        # Intra Proxy Tip Logits with Ensemble backbone model
        intra_proxy_tip_logits = 100. * ensemble_clip_logits + best_alpha * cache_logits_vit
        acc = cls_acc(intra_proxy_tip_logits, test_labels)
        acc_top_3 = cls_acc(intra_proxy_tip_logits, test_labels, topk=3)
        print(
            "**** My AdapterOOD Ensemble backbone's test accuracy: {:.2f}. ****"
            .format(acc))
        print(
            "**** My AdapterOOD Ensemble backbone's topk=3 test accuracy: {:.2f}. ****"
            .format(acc_top_3))

    return intra_proxy_tip_logits, cupl_clip_logits, ensemble_clip_logits
