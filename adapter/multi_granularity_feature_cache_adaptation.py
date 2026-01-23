# multi_granularity_feature_cache_adaptation.py
# This script implements a multi_granularity_feature_cache_adaptation for remote sensing datasets using CLIP models.
# It compares single backbone (ViT) and multi-backbone (ViT + ResNet) performance across various shots.
# Parameters are loaded from a YAML configuration file.

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Load configuration from YAML file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Resolve template references in datasets
for ds_name, ds_cfg in config['datasets'].items():
    if isinstance(ds_cfg['templates'], str):
        ref_ds = ds_cfg['templates']
        ds_cfg['templates'] = config['datasets'][ref_ds]['templates']

datasets_configs = config['datasets']
hyperparams = config['hyperparameters']
shots_list = hyperparams['shots_list']
alpha = hyperparams['alpha']
beta = hyperparams['beta']
gamma = hyperparams['gamma']
alpha_cache = hyperparams['alpha_cache']
batch_size = hyperparams['batch_size']
vit_model = hyperparams['vit_model']
resnet_model = hyperparams['resnet_model']

# Function to create data loaders for training and testing
def get_dataloader(ds_config, batch_size=batch_size):
    """
    Create train and test data loaders for a given dataset configuration.
    
    Args:
    ds_config (dict): Dataset configuration including path.
    batch_size (int): Batch size for data loaders.
    
    Returns:
    tuple: Train and test data loaders.
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821079), (0.26862954, 0.26130258, 0.27577711))
    ])
    dataset = ImageFolder(ds_config['path'], transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Function for single backbone Tip-Adapter
def run_single_tip_adapter(model, train_loader, test_loader, ds_config, shots=1, alpha=alpha, beta=beta):
    """
    Run Tip-Adapter with a single backbone (ViT).
    
    Args:
    model: CLIP model.
    train_loader: Training data loader.
    test_loader: Testing data loader.
    ds_config (dict): Dataset configuration.
    shots (int): Number of shots per class.
    alpha (float): Scaling factor for cache logits.
    beta (float): Scaling factor for affinity.
    
    Returns:
    float: Accuracy percentage.
    """
    device = next(model.parameters()).device
    
    # Extract text features (Zero-shot Classifier)
    print("Extracting text features for single backbone...")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in ds_config['classes']:
            texts = [template.format(classname) for template in ds_config['templates']]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    # Build Cache Model
    print(f"Constructing single backbone cache with {shots}-shot...")
    cache_keys = []
    cache_values = []
    
    with torch.no_grad():
        class_counts = {i: 0 for i in range(len(ds_config['classes']))}
        for i, (images, target) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            target = target.to(device)
            
            mask = torch.tensor([class_counts[t.item()] < shots for t in target], device=device)
            if not mask.any(): continue
            
            features = model.encode_image(images[mask])
            features /= features.norm(dim=-1, keepdim=True)
            
            cache_keys.append(features)
            cache_values.append(F.one_hot(target[mask], num_classes=len(ds_config['classes'])))
            
            for t in target[mask]:
                class_counts[t.item()] += 1
            if all(count >= shots for count in class_counts.values()):
                break

    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)  # [D, N*shots]
    cache_values = torch.cat(cache_values, dim=0).to(device=device, dtype=cache_keys.dtype)   # [N*shots, N]

    # Testing
    print("Testing single backbone Tip-Adapter...")
    correct, total = 0, 0
    with torch.no_grad():
        for images, target in tqdm(test_loader):
            images, target = images.to(device), target.to(device)
            
            test_features = model.encode_image(images)
            test_features /= test_features.norm(dim=-1, keepdim=True)
            
            logits_zs = test_features @ zeroshot_weights
            
            affinity = test_features @ cache_keys
            cache_logits = ((-beta * (1 - affinity)).exp()) @ cache_values
            
            tip_logits = logits_zs + cache_logits * alpha
            
            pred = tip_logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    acc = 100 * correct / total
    print(f"\nSingle Backbone Final Accuracy: {acc:.2f}%")
    return acc

# Class for multi_granularity_feature_cache_adaptation
class MultiBackboneCacheAdapter:
    def __init__(self, device="cuda"):
        """
        Initialize multi_granularity_feature_cache_adaptation with ViT and ResNet models.
        
        Args:
        device (str): Device to use ('cuda' or 'cpu').
        """
        self.device = device
        print("Loading ViT and ResNet CLIP models...")
        self.model_vit, _ = clip.load(vit_model, device=device)
        self.model_resnet, _ = clip.load(resnet_model, device=device)
    
    def extract_cache_features(self, loader, ds_config, shots=1):
        """
        Extract features from multi-backbone models for cache.
        
        Args:
        loader: Data loader.
        ds_config (dict): Dataset configuration.
        shots (int): Number of shots per class.
        
        Returns:
        tuple: ViT features, ResNet features, pseudo labels.
        """
        print(f"Extracting multi-backbone features (shots={shots})...")
        f_vit_list, f_res_list, labels_list = [], [], []
        class_counts = {i: 0 for i in range(len(ds_config['classes']))}
        
        with torch.no_grad():
            for images, target in tqdm(loader):
                images = images.to(self.device)
                target = target.to(self.device)
                
                mask = torch.tensor([class_counts[t.item()] < shots for t in target], device=self.device)
                if not mask.any(): continue
                
                feat_vit = self.model_vit.encode_image(images[mask])
                feat_vit /= feat_vit.norm(dim=-1, keepdim=True)
                
                feat_res = self.model_resnet.encode_image(images[mask])
                feat_res /= feat_res.norm(dim=-1, keepdim=True)
                
                f_vit_list.append(feat_vit)
                f_res_list.append(feat_res)
                labels_list.append(target[mask])
                
                for t in target[mask]: class_counts[t.item()] += 1
                if all(c >= shots for c in class_counts.values()): break
        
        F_vit = torch.cat(f_vit_list, dim=0)
        F_resnet = torch.cat(f_res_list, dim=0)
        L_pseudo = F.one_hot(torch.cat(labels_list, dim=0), len(ds_config['classes'])).to(device=self.device, dtype=F_vit.dtype)
        
        return F_vit, F_resnet, L_pseudo
    
    def get_zeroshot_weights(self, ds_config):
        """
        Get zero-shot text weights using ViT model.
        
        Args:
        ds_config (dict): Dataset configuration.
        
        Returns:
        torch.Tensor: Zero-shot weights.
        """
        with torch.no_grad():
            weights = []
            for classname in ds_config['classes']:
                texts = clip.tokenize([t.format(classname) for t in ds_config['templates']]).to(self.device)
                emb = self.model_vit.encode_text(texts)
                emb /= emb.norm(dim=-1, keepdim=True)
                weights.append(emb.mean(dim=0) / emb.mean(dim=0).norm())
        return torch.stack(weights, dim=1)
    
    def inference(self, test_loader, F_vit, F_resnet, L_pseudo, W_t, beta=beta, gamma=gamma, alpha_cache=alpha_cache):
        """
        Perform inference with multi_granularity_feature_cache_adaptation.
        
        Args:
        test_loader: Testing data loader.
        F_vit: ViT cache features.
        F_resnet: ResNet cache features.
        L_pseudo: Pseudo labels.
        W_t: Zero-shot weights.
        beta (float): Scaling for affinity.
        gamma (float): Weight for ResNet affinity.
        alpha_cache (float): Scaling for cache logits.
        
        Returns:
        float: Accuracy percentage.
        """
        self.model_vit.eval()
        self.model_resnet.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, target in tqdm(test_loader):
                images, target = images.to(self.device), target.to(self.device)
                
                f_vit_test = self.model_vit.encode_image(images)
                f_vit_test /= f_vit_test.norm(dim=-1, keepdim=True)
                
                f_res_test = self.model_resnet.encode_image(images)
                f_res_test /= f_res_test.norm(dim=-1, keepdim=True)
                
                A_res = torch.exp(beta * (f_res_test @ F_resnet.T - 1))
                A_vit = torch.exp(beta * (f_vit_test @ F_vit.T - 1))
                
                A_cache = A_vit + gamma * A_res
                
                logits_zs = f_vit_test @ W_t
                M_cache = alpha_cache * (A_cache @ L_pseudo) + logits_zs
                
                acc = (M_cache.argmax(dim=1) == target).sum().item()
                correct += acc
                total += target.size(0)
        
        acc = 100 * correct / total
        print(f"\nMulti Backbone Final Accuracy: {acc:.2f}%")
        return acc

# Main function to run the comparison
def main():
    """
    Main function to run comparisons across shots and datasets, and plot results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_single, _ = clip.load(vit_model, device=device)
    adapter_multi = MultiBackboneCacheAdapter(device=device)
    
    single_avgs = []
    multi_avgs = []
    
    for shots in shots_list:
        single_accs = []
        multi_accs = []
        for ds_name, ds_cfg in datasets_configs.items():
            print(f"\nProcessing {ds_cfg['path']} with {shots}-shot")
            train_loader, test_loader = get_dataloader(ds_cfg)
            acc_single = run_single_tip_adapter(model_single, train_loader, test_loader, ds_cfg, shots=shots)
            single_accs.append(acc_single)
            F_v, F_r, L_p = adapter_multi.extract_cache_features(train_loader, ds_cfg, shots=shots)
            W_t = adapter_multi.get_zeroshot_weights(ds_cfg)
            acc_multi = adapter_multi.inference(test_loader, F_v, F_r, L_p, W_t)
            multi_accs.append(acc_multi)
        avg_single = np.mean(single_accs)
        avg_multi = np.mean(multi_accs)
        single_avgs.append(avg_single)
        multi_avgs.append(avg_multi)
        print(f"\nFor {shots}-shot: Single Avg {avg_single:.2f}%, Multi Avg {avg_multi:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(shots_list, single_avgs, marker='o', label='Tip-Adapter')
    plt.plot(shots_list, multi_avgs, marker='o', label='Proposed')
    plt.xlabel('Shots')
    plt.ylabel('Avg Accuracy (%)')
    plt.title('Proposed vs Tip-Adapter on RS Datasets')
    plt.legend()
    plt.grid(True)
    plt.savefig('shots_comparison.png')

if __name__ == "__main__":
    main()