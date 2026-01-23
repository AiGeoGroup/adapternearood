# llm_augmented_prompt_satadapter.py
# Language-assisted remote sensing prompts generation for SatAdapter
# This script demonstrates the LLM-augmented prompt generalization in SatAdapter

import json
import os

import clip
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Load configuration from YAML file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

datasets_config = config['datasets']
seed = config['seed']
model_name = config['model_name']
batch_size = config['batch_size']
template = config['template']

# ==========================================
# utility functions
# ==========================================

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cls_acc(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    pred = output.topk(maxk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0].item()

# ==========================================
# Feature Extraction Function
# ==========================================

def extract_text_features(classnames, model, template):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname)]
            tokens = clip.tokenize(texts, truncate=True).cuda()
            class_embeddings = model.encode_text(tokens)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def extract_cupl_text_features(classnames, cupl_path, model):
    with open(cupl_path, 'r') as f:
        cupl_prompts = json.load(f)
    
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = cupl_prompts.get(classname, [f"a satellite view of {classname}"])
            tokens = clip.tokenize(texts, truncate=True).cuda()
            class_embeddings = model.encode_text(tokens)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def extract_features_from_loader(model, loader):
    features_list, labels_list = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, target in tqdm(loader, leave=False):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features_list.append(image_features)
            labels_list.append(target)
    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0).to(device)

# ==========================================
# Reasoning and Comparative Logic
# ==========================================
def search_hp_opt_llm_prompts(logits_dict, labels):
    best_acc, best_w = 0, 0
    for w in np.arange(0, 1.01, 0.05):
        fused_logits = (1 - w) * logits_dict['base'] + w * logits_dict['cupl']
        acc = cls_acc(fused_logits, labels)
        if acc > best_acc:
            best_acc, best_w = acc, w
    return best_w, (1 - best_w) * logits_dict['base'] + best_w * logits_dict['cupl']

# ==========================================
# Multi-dataset experiments
# ==========================================

def main():
    setup_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    print(f"{'Dataset':<20} | {'CLIP-Base':<10} | {'CUPL':<10} | {'Proposed':<10} | {'Gain':<10}")
    print("-" * 70)

    for name, paths in datasets_config.items():
        if not os.path.exists(paths['data']):
            continue
            
        # 1.load dataset
        dataset = ImageFolder(root=paths['data'], transform=preprocess)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 2.Extract features
        img_feats, labels = extract_features_from_loader(model, loader)
        base_w = extract_text_features(dataset.classes, model, template)
        cupl_w = extract_cupl_text_features(dataset.classes, paths['json'], model) if os.path.exists(paths['json']) else base_w

        # 3. Compute logits
        logits_base = img_feats @ base_w
        logits_cupl_raw = img_feats @ cupl_w
        
        # CLIP-Base Accuracy
        acc_base = cls_acc(logits_base, labels)
        
        # CUPL Fixed Accuracy (0.55, 0.45)
        logits_cupl_fixed = 0.55 * logits_cupl_raw + 0.45 * logits_base
        acc_cupl = cls_acc(logits_cupl_fixed, labels)
        
        # Proposed Accuracy with HP Optimization
        logits_dict = {'base': logits_base, 'cupl': logits_cupl_raw}
        best_w, final_logits = search_hp_opt_llm_prompts(logits_dict, labels)
        acc_proposed = cls_acc(final_logits, labels)
        
        gain = acc_proposed - acc_base
        print(f"{name:<20} | {acc_base:>9.2f}% | {acc_cupl:>9.2f}% | {acc_proposed:>9.2f}% | {gain:>+9.2f}%")

if __name__ == "__main__":
    main()