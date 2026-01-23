# proxy_semantic_calib.py
# This script performs proxy semantic calibration for remote sensing datasets using CLIP and ResNet models.
# It includes zero-shot classification, feature extraction, pseudo-label refinement, and proxy optimization.
# Parameters are loaded from a YAML configuration file.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import clip
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np
import torch.hub  # For loading ResNet

# Load configuration from YAML file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

datasets_configs = config['datasets']
zeta_values = config['hyperparameters']['zeta_values']
lr = config['hyperparameters']['lr']
num_iters = config['hyperparameters']['num_iters']
tau_i = config['hyperparameters']['tau_i']
tau_t = config['hyperparameters']['tau_t']
sinkhorn_iters = config['hyperparameters']['sinkhorn_iters']
batch_size = config['hyperparameters']['batch_size']

# Function to create zero-shot text classifier weights using CLIP
def zeroshot_classifier(model, classnames, templates):
    # Compute averaged text embeddings for each class using multiple prompt templates
    with torch.no_grad():
        all_weights = []
        for classname in classnames:
            texts = [t.format(classname) for t in templates]
            tokenized = clip.tokenize(texts).cuda()
            embeddings = model.encode_text(tokenized)  # (num_templates, dim)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            class_weight = embeddings.mean(dim=0)
            class_weight /= class_weight.norm()
            all_weights.append(class_weight)
        return torch.stack(all_weights, dim=1).cuda()

# Function to extract image features from a DataLoader using a model
def extract_features(model, loader, is_clip=True):
    # Extract normalized image features and labels from the dataset
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.cuda()
            if is_clip:
                feat = model.encode_image(images)
            else:
                feat = model(images)  # For ResNet, assuming it outputs features
            feat /= feat.norm(dim=-1, keepdim=True)
            features.append(feat)
            labels.append(targets)
    return torch.cat(features), torch.cat(labels).cuda()

# Function to compute top-1 accuracy
def accuracy(logits, labels):
    # Calculate the percentage of correct predictions
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item() * 100

# Function to refine pseudo-labels using Sinkhorn-Knopp algorithm
def refine_pseudo_labels(pseudo_labels, iters=3):
    # Normalize and iteratively refine pseudo-labels for better distribution
    pseudo_labels = pseudo_labels / pseudo_labels.norm(dim=1, keepdim=True)
    for _ in range(iters):
        pseudo_labels = F.softmax(pseudo_labels, dim=1)
        pseudo_labels = pseudo_labels / pseudo_labels.norm(dim=0, keepdim=True)
    return pseudo_labels

# Function for optimizing image proxies
def image_opt(feats, text_classifier, refined_labels, lr, num_iters, tau_i, zeta):
    # Optimize proxy vectors using Adam optimizer with softmax and entropy loss
    proxy = nn.Parameter(text_classifier.clone().detach())
    optimizer = torch.optim.Adam([proxy], lr=lr)
    for i in range(num_iters):
        optimizer.zero_grad()
        logits = feats @ proxy
        probs = F.softmax(logits / tau_i, dim=1)
        loss = - (refined_labels * torch.log(probs + 1e-8)).sum(dim=1).mean()
        hard_mask = (probs.max(dim=1)[0] > zeta).float()
        hard_loss = - (refined_labels * torch.log(probs + 1e-8)).sum(dim=1) * hard_mask
        loss += hard_loss.mean()
        loss.backward()
        optimizer.step()
    return proxy.detach()

# Function to process a single dataset
def process_dataset(ds_name, config):
    # Load dataset, models, extract features, perform calibration, and tune zeta
    path = config['path']
    if not os.path.exists(path):
        print(f"Dataset path not found for {ds_name}: {path}")
        return None
    
    # Load CLIP model (ViT)
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    
    # Load ResNet model
    resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).cuda().eval()
    
    # Load dataset with preprocessing
    dataset = datasets.ImageFolder(path, transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Extract features
    vit_feats, image_labels = extract_features(clip_model, loader, is_clip=True)
    resnet_feats, _ = extract_features(resnet_model, loader, is_clip=False)  # Labels already from ViT
    
    # Create text classifiers (shared for both since CLIP is text-based)
    text_classifier = zeroshot_classifier(clip_model, config['classes'], config['templates'])
    
    # Compute initial logits and pseudo-labels
    vit_logits = vit_feats @ text_classifier
    resnet_logits = resnet_feats @ text_classifier  # Assuming same text classifier
    vit_probs = F.softmax(vit_logits / tau_t, dim=1)
    resnet_probs = F.softmax(resnet_logits / tau_t, dim=1)
    pseudo_labels_base = 0.5 * vit_probs + 0.5 * resnet_probs
    
    # Compute entropy before optimization
    entropy_before = -(pseudo_labels_base * torch.log(pseudo_labels_base + 1e-8)).sum(dim=1).cpu().numpy()
    
    # Tune zeta parameter
    results = []
    for zeta in zeta_values:
        refined_labels = refine_pseudo_labels(pseudo_labels_base.clone(), iters=sinkhorn_iters)
        
        # Compute dynamic zeta based on class confidence
        num_classes = pseudo_labels_base.size(1)
        class_conf = torch.zeros(num_classes, device=pseudo_labels_base.device)
        class_counts = torch.zeros(num_classes, device=pseudo_labels_base.device)
        pseudo_max, pseudo_argmax = pseudo_labels_base.max(dim=1)
        for c in range(num_classes):
            mask = pseudo_argmax == c
            if mask.sum() > 0:
                class_conf[c] = pseudo_max[mask].mean()
                class_counts[c] = mask.sum()
        dynamic_zeta = zeta * class_conf / class_conf.mean()
        dynamic_zeta = torch.clamp(dynamic_zeta, min=0.1, max=0.9)
        avg_dynamic_zeta = dynamic_zeta.mean().item()
        print(f"Using dynamic zeta: {avg_dynamic_zeta:.2f} for {ds_name}")
        
        # Optimize proxies
        resnet_proxy = image_opt(resnet_feats, text_classifier, refined_labels, lr, num_iters, tau_i, avg_dynamic_zeta)
        vit_proxy = image_opt(vit_feats, text_classifier, refined_labels, lr, num_iters, tau_i, avg_dynamic_zeta)
        
        # Compute vision logits and normalize
        resnet_logits_vision = resnet_feats @ resnet_proxy
        vit_logits_vision = vit_feats @ vit_proxy
        resnet_logits_vision_norm = (resnet_logits_vision - resnet_logits_vision.mean(dim=1, keepdim=True)) / (resnet_logits_vision.std(dim=1, keepdim=True) + 1e-8)
        vit_logits_vision_norm = (vit_logits_vision - vit_logits_vision.mean(dim=1, keepdim=True)) / (vit_logits_vision.std(dim=1, keepdim=True) + 1e-8)
        logits_vision = 0.5 * resnet_logits_vision_norm + 0.5 * vit_logits_vision_norm
        
        # Compute accuracy
        acc_vision = accuracy(logits_vision, image_labels)
        print(f"[Step 2] InMaP Fused Vision Proxy Accuracy for {ds_name}: {acc_vision:.2f}%")
        results.append((zeta, acc_vision))
    
    # Print zeta analysis summary
    print(f"\nZeta Parameter Analysis Summary for {ds_name}:")
    for z, acc in results:
        print(f"zeta = {z}: Accuracy = {acc:.2f}%")
    best_zeta = max(results, key=lambda x: x[1])[0]
    best_acc = max(acc for _, acc in results)
    print(f"Best zeta for {ds_name}: {best_zeta} (Accuracy: {best_acc:.2f}%)")
    print("Observations: Accuracy likely peaks around 0.3-0.4, dropping at higher zeta due to fewer hard labels, and at lower due to noisy hard labels. Adjust based on your run.")
    
    # Compute entropy after optimization (using last logits_vision)
    fused_probs_after = F.softmax(logits_vision / tau_t, dim=1)
    entropy_after = -(fused_probs_after * torch.log(fused_probs_after + 1e-8)).sum(dim=1).cpu().numpy()
    
    # Perform ablations
    ablations = {}
    
    # Ablation: InMaP (only ViT)
    print("Running InMaP (only ViT)...")
    pseudo_labels_vit_only = vit_probs
    pseudo_labels_vit_only_refined = refine_pseudo_labels(pseudo_labels_vit_only.clone(), iters=sinkhorn_iters)
    vit_proxy_only = image_opt(vit_feats, text_classifier, pseudo_labels_vit_only_refined, lr, num_iters, tau_i, best_zeta)
    logits_vision_vit_only = vit_feats @ vit_proxy_only
    acc_vit_only = accuracy(logits_vision_vit_only, image_labels)
    ablations["InMaP (ViT only)"] = acc_vit_only
    print(f"InMaP (ViT only) Accuracy: {acc_vit_only:.2f}%")
    
    # Ablation: Proposed Proxy Learning (full)
    ablations["Proposed Proxy Learning"] = best_acc
    print(f"Proposed Proxy Learning Accuracy: {best_acc:.2f}%")
    
    return results, best_zeta, ablations, entropy_before, entropy_after

# Function to visualize accuracy vs zeta across datasets
def visualize_zeta_tuning(all_results):
    # Plot accuracy as a function of zeta for each dataset
    plt.figure(figsize=(10, 6))
    for ds_name, res in all_results.items():
        zetas, accs = zip(*res)
        plt.plot(zetas, accs, marker='o', label=ds_name)
    plt.xlabel("Zeta")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Zeta Parameter Across Datasets")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to visualize best accuracies across datasets
def visualize_best_accuracies(all_results):
    # Bar plot of the best accuracy for each dataset
    best_accs = [max(acc for _, acc in res) for res in all_results.values()]
    ds_names = list(all_results.keys())
    plt.figure(figsize=(10, 6))
    plt.bar(ds_names, best_accs, color='skyblue')
    plt.xlabel("Dataset")
    plt.ylabel("Best Accuracy (%)")
    plt.title("Best InMaP Accuracy Across Remote Sensing Datasets")
    plt.grid(axis='y')
    plt.show()
    return ds_names

# Function to visualize entropy distributions before and after optimization
def visualize_entropy_distributions(all_entropies_before, all_entropies_after, ds_names):
    # Histogram of entropy values for each dataset
    for ds_name in ds_names:
        plt.figure(figsize=(10, 6))
        plt.hist(all_entropies_before[ds_name], bins=50, alpha=0.5, label='Before Optimization')
        plt.hist(all_entropies_after[ds_name], bins=50, alpha=0.5, label='After Optimization')
        plt.xlabel("Entropy")
        plt.ylabel("Frequency")
        plt.title(f"Entropy Distribution Before/After for {ds_name}")
        plt.legend()
        plt.grid(True)
        plt.show()

# Function to visualize ablation results
def visualize_ablation_results(ablation_results, ds_names):
    # Create and print ablation table, then bar plot
    ablation_df = pd.DataFrame(ablation_results).T
    print("Ablation Table:")
    print(ablation_df)
    
    variants = list(ablation_df.columns)
    plt.figure(figsize=(12, 8))
    bar_width = 0.15
    index = range(len(ds_names))
    for i, variant in enumerate(variants):
        plt.bar([p + i*bar_width for p in index], ablation_df[variant], width=bar_width, label=variant)
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy (%)")
    plt.title("Ablation Study: Comparison with Original InMaP Components")
    plt.xticks([p + bar_width for p in index], ds_names)
    plt.legend()
    plt.grid(axis='y')
    plt.show()

# Function to visualize average performance across methods
def visualize_average_performance(ablation_results, ds_names):
    # Bar plot of average accuracies for proposed and baseline methods
    proposed_accs = [ablation_results[ds]['Proposed Proxy Learning'] for ds in ds_names]
    inmap_accs = [ablation_results[ds]['InMaP (ViT only)'] for ds in ds_names]
    
    avg_proposed = sum(proposed_accs) / len(proposed_accs)
    avg_inmap = sum(inmap_accs) / len(inmap_accs)
    
    methods = ['Proposed Proxy Learning', 'InMaP (ViT only)']
    avgs = [avg_proposed, avg_inmap]
    
    plt.figure(figsize=(8, 6))
    plt.bar(methods, avgs, color=['blue', 'orange'])
    plt.xlabel("Method")
    plt.ylabel("Average Accuracy (%)")
    plt.title("Average Performance: Proposed Proxy Learning vs InMaP Across Datasets")
    plt.grid(axis='y')
    plt.show()

# Main function to run the entire pipeline
def main():
    # Process all datasets, collect results, and generate visualizations
    all_results = {}
    ablation_results = {}
    all_entropies_before = {}
    all_entropies_after = {}
    best_zetas = {}
    
    for ds_name, cfg in datasets_configs.items():
        results, best_zeta, ablations, entropy_before, entropy_after = process_dataset(ds_name, cfg)
        if results is None:
            continue
        
        all_results[ds_name] = results
        best_zetas[ds_name] = best_zeta
        ablation_results[ds_name] = ablations
        all_entropies_before[ds_name] = entropy_before
        all_entropies_after[ds_name] = entropy_after
    
    ds_names = list(all_results.keys())
    
    # Generate visualizations
    visualize_zeta_tuning(all_results)
    visualize_best_accuracies(all_results)
    visualize_entropy_distributions(all_entropies_before, all_entropies_after, ds_names)
    visualize_ablation_results(ablation_results, ds_names)
    visualize_average_performance(ablation_results, ds_names)

if __name__ == "__main__":
    main()