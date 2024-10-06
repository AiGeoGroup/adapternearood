import torch
import torch.nn.functional as F

from adapter.libs.datautils import *

# -------------------------------------------------------------------------
# Intra-Modal Proxy Learning CLIP
# -------------------------------------------------------------------------
def image_opt(feat,
              init_classifier,
              plabel,
              lr=10,
              iter=2000,
              tau_i=0.04,
              alpha=0.6):
    ins, dim = feat.shape  # feat: feature shape
    val, idx = torch.max(plabel, dim=1)  # plabel: pseudo label
    mask = val > alpha
    #print(ins, dim)
    #print(val.shape, idx.shape)
    #print(mask.shape)
    plabel[mask, :] = 0
    #print(plabel)
    base = feat.T @ plabel
    # print(base.shape)
    # print(base)
    classifier = init_classifier.clone()
    pre_norm = float('inf')
    for i in range(0, iter):
        prob = F.softmax(feat @ classifier / tau_i, dim=1)
        grad = feat.T @ prob - base
        temp = torch.norm(grad)
        if temp > pre_norm:
            lr /= 2.
        pre_norm = temp
        classifier -= (lr / (ins * tau_i)) * grad
        classifier = F.normalize(classifier, dim=0)
    return classifier


def sinkhorn(M, tau_t=0.01, gamma=0, iter=20):
    row, col = M.shape
    P = F.softmax(M / tau_t, dim=1)
    P /= row
    if gamma > 0:
        q = torch.sum(P, dim=0, keepdim=True)
        q = q**gamma
        q /= torch.sum(q)
    for it in range(0, iter):
        # total weight per column must be 1/col or q_j
        P /= torch.sum(P, dim=0, keepdim=True)
        if gamma > 0:
            P *= q
        else:
            P /= col
        # total weight per row must be 1/row
        P /= torch.sum(P, dim=1, keepdim=True)
        P /= row
    P *= row  # keep each row sum to 1 as the pseudo label
    return P


### Intra-Modal Proxy Learning CLIP
def run_intra_model_proxy_learning_clip(configs, logits_base, cupl_clip_weights, val_features, val_labels):
    print('obtain refined labels by Sinkhorn distance')
    logits_t = logits_base
    plabel = sinkhorn(logits_t, configs['tau_t'], configs['gamma'], configs['iters_sinkhorn'])

    print('obtain vision proxy with Sinkhorn distance')
    # text_classifier = clip_weights_template
    text_classifier = cupl_clip_weights
    # text_classifier = zeroshot_weights_both
    image_feat = val_features
    image_classifier = image_opt(image_feat, text_classifier, plabel, configs['lr'], configs['iters_proxy'], configs['tau_i'], configs['alpha'])
    logits_i = 100.0 * image_feat @ image_classifier

    acc_logits_i = cls_acc(logits_i, val_labels)
    acc_logits_i_3 = cls_acc(logits_i, val_labels, topk=3)
    print(acc_logits_i)
    print(acc_logits_i_3)
    return image_classifier, plabel


# 创建pseudo labels、init_pse_cache_keys、init_pse_cache_values
def init_pselabel_cache(the_features, the_clip_weights, test_features, class_num, cache_k_shot=1):
        # class_num = num_classes # the_clip_weights.shape[1]
        logits  = the_features @ the_clip_weights
        best_scores, best_class_id = torch.max(logits, dim=1)
        init_pse_cache_keys = []
        init_pse_cache_values = []

        for class_id in range(class_num):
            class_positions = best_class_id == class_id
            pred_len = int(torch.sum(class_positions))
            if pred_len > 0:
                class_examples_scores = best_scores * class_positions
                _, good_examples = torch.topk(class_examples_scores, k=min(cache_k_shot, pred_len))
                test_features_values = torch.zeros([len(good_examples), class_num]).cuda()
                test_features_values[:, class_id] = 1

                init_pse_cache_keys.append(test_features[good_examples])
                init_pse_cache_values.append(test_features_values)
        init_pse_cache_keys = torch.cat(init_pse_cache_keys, 0)
        init_pse_cache_values = torch.cat(init_pse_cache_values, 0).half()

        return init_pse_cache_keys, init_pse_cache_values