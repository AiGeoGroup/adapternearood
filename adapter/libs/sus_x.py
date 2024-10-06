import torch.nn as nn
import torch

from adapter.libs.datautils import *

def compute_image_text_distributions(temp, few_shot_images_features_agg, test_features, val_features, vanilla_zeroshot_weights):
    few_shot_image_class_distribution = few_shot_images_features_agg.T @ vanilla_zeroshot_weights  # vanilla_zeroshot_weights
    few_shot_image_class_distribution = nn.Softmax(dim=-1)(few_shot_image_class_distribution/temp) # temp调节温度

    test_image_class_distribution = test_features @ vanilla_zeroshot_weights
    test_image_class_distribution = nn.Softmax(dim=-1)(test_image_class_distribution/temp)

    val_image_class_distribution = val_features @ vanilla_zeroshot_weights
    val_image_class_distribution = nn.Softmax(dim=-1)(val_image_class_distribution/temp)

    return few_shot_image_class_distribution, test_image_class_distribution, val_image_class_distribution


def get_kl_div_sims(cfgs, test_features, val_features, few_shot_features, clip_weights):

    few_shot_image_class_distribution, test_image_class_distribution, val_image_class_distribution = compute_image_text_distributions(cfgs['susx_temp'], few_shot_features, test_features, val_features, clip_weights)

    few_shot_kl_divs_sim = get_kl_divergence_sims(few_shot_image_class_distribution, few_shot_image_class_distribution)
    test_kl_divs_sim = get_kl_divergence_sims(few_shot_image_class_distribution, test_image_class_distribution)
    val_kl_divs_sim = get_kl_divergence_sims(few_shot_image_class_distribution, val_image_class_distribution)

    return few_shot_kl_divs_sim, test_kl_divs_sim, val_kl_divs_sim

def get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution):
    bs = 100
    kl_divs_sim = torch.zeros((test_image_class_distribution.shape[0], train_image_class_distribution.shape[0]))

    for i in range(test_image_class_distribution.shape[0]//bs):
        curr_batch = test_image_class_distribution[i*bs : (i+1)*bs]
        repeated_batch = torch.repeat_interleave(curr_batch, train_image_class_distribution.shape[0], dim=0)
        q = train_image_class_distribution
        q_repeated = torch.cat([q]*bs)
        kl = repeated_batch * (repeated_batch.log() - q_repeated.log())
        kl = kl.sum(dim=-1)
        kl = kl.view(bs, -1)
        kl_divs_sim[ i*bs : (i+1)*bs , : ] = kl

    return kl_divs_sim

def scale_(x, target):

    y = (x - x.min()) / (x.max() - x.min())
    y *= target.max() - target.min()
    y += target.min()

    return y

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def run_sus_x_adapter(val_features, val_labels, test_features, test_labels, train_images_features_agg, train_images_targets, zeroshot_weights, val_kl_divs_sim, test_kl_divs_sim):
    '''
    train_images_features_agg: few_shot_images_features_agg
    train_images_targets: few_shot_image_labels
    '''
    search_scale = [50, 50, 30]
    search_step = [200, 20, 50]

    alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in range(search_step[1])]
    beta_list = [i * (search_scale[0] - 1) / search_step[0] + 1 for i in range(search_step[0])]
    gamma_list = [i * (search_scale[2] - 0.1) / search_step[2] + 0.1 for i in range(search_step[2])]

    best_tipx_acc = 0

    best_gamma_tipx, best_alpha_tipx, best_beta_tipx = 0, 0, 0

    for alpha in alpha_list:
        for beta in beta_list:
            n = 0.
            batch_idx = 0

            new_knowledge = val_features @ train_images_features_agg
            cache_logits = ((-1) * (beta - beta * new_knowledge)).exp() @ (train_images_targets)
            clip_logits = 100. * val_features @ zeroshot_weights

            batch_idx += 1
            n += val_features.size(0)

            neg_affs = scale_((val_kl_divs_sim).cuda(), new_knowledge)
            affinities = -neg_affs
            kl_logits = affinities.half() @ train_images_targets

            for gamma in gamma_list:
                tipx_top1, tipx_top5 = 0., 0.

                tipx_logits = clip_logits + kl_logits * gamma + cache_logits * alpha
                tipx_acc1, tipx_acc5 = accuracy(tipx_logits, val_labels, topk=(1, 5))
                tipx_top1 += tipx_acc1
                tipx_top5 += tipx_acc5
                tipx_top1 = (tipx_top1 / n) * 100
                tipx_top5 = (tipx_top5 / n) * 100

                if tipx_top1 > best_tipx_acc:
                    best_tipx_acc = tipx_top1
                    best_alpha_tipx = alpha
                    best_gamma_tipx = gamma
                    best_beta_tipx = beta

    n = test_features.size(0)

    clip_logits = 100. * test_features @ zeroshot_weights

    neg_affs = scale_((test_kl_divs_sim).cuda(), new_knowledge)
    affinities = -neg_affs
    kl_logits = affinities.half() @ train_images_targets

    tipx_top1, tipx_top5 = 0., 0.

    new_knowledge = test_features @ train_images_features_agg
    cache_logits = ((-1) * (best_beta_tipx - best_beta_tipx * new_knowledge)).exp() @ train_images_targets
    sus_x_logits = clip_logits + kl_logits * best_gamma_tipx + cache_logits * best_alpha_tipx

    acc_top_1 = cls_acc(sus_x_logits, test_labels)
    acc_top_3 = cls_acc(sus_x_logits, test_labels, topk=3)
    print("**** SuS-X-Adapter's val accuracy: {:.2f}. ***".format(acc_top_1))
    print("**** SuS-X-Adapter's top-3 val accuracy: {:.2f}. ***".format(acc_top_3))

    return tipx_top1, best_alpha_tipx, best_beta_tipx, best_gamma_tipx, sus_x_logits
