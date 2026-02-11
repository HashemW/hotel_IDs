import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from pytorch_metric_learning import losses, miners
import os
import glob
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import defaultdict
import random

# --- CONFIG ---
# Update this path if needed
FEATURES_DIR = r"D:\Coding\hotels\grid_features\train_images" 

NUM_EPOCHS = 30
BATCH_SIZE = 32      
INPUT_DIM = 1024       
PCA_DIM = 256          
NUM_CLUSTERS = 64      
LEARNING_RATE = 0.0001 
MARGIN = 0.2           
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. ROBUST BALANCED SAMPLER ---
class BalancedBatchSampler(Sampler):
    """
    Fixed to work with PyTorch Subsets (Train/Val splits).
    Ensures we only sample indices that exist in the current split.
    """
    def __init__(self, dataset, n_classes=8, n_samples=4):
        self.labels = []
        self.dataset = dataset
        self.label_to_indices = defaultdict(list)
        
        # 1. Resolve correct labels based on whether it's a Subset or full Dataset
        if isinstance(dataset, Subset):
            # It's a split (Train set)
            indices = dataset.indices # Global indices in the parent
            parent = dataset.dataset
            
            # Map Local Index (0..N) -> Label
            for local_idx, global_idx in enumerate(indices):
                # We need to find the label without loading the file (slow)
                # We rely on the parent's file list
                file_path = parent.files[global_idx]
                cls_name = os.path.basename(os.path.dirname(file_path))
                label = parent.class_to_idx[cls_name]
                
                self.labels.append(label)
                self.label_to_indices[label].append(local_idx)
        else:
            # It's the full dataset (no split)
            for local_idx, f in enumerate(dataset.files):
                cls_name = os.path.basename(os.path.dirname(f))
                label = dataset.class_to_idx[cls_name]
                self.labels.append(label)
                self.label_to_indices[label].append(local_idx)
            
        # 2. Filter classes that might have 0 samples in this split
        self.labels_set = [l for l in self.label_to_indices.keys() if len(self.label_to_indices[l]) > 0]
            
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_classes * self.n_samples
        
        # Determine number of batches
        if len(self.labels_set) < n_classes:
            self.n_batches = 1 # Not enough classes for even one batch, risky but avoids crash
        else:
            self.n_batches = len(self.labels) // self.batch_size

    def __iter__(self):
        count = 0
        while count < self.n_batches:
            # 1. Pick P random classes
            # Ensure we don't crash if we have fewer classes than requested
            curr_n_classes = min(len(self.labels_set), self.n_classes)
            classes = np.random.choice(self.labels_set, curr_n_classes, replace=False)
            
            batch_indices = []
            
            for class_ in classes:
                indices = self.label_to_indices[class_]
                # 2. Pick K random images from each class
                if len(indices) >= self.n_samples:
                    chosen = np.random.choice(indices, self.n_samples, replace=False)
                else:
                    chosen = np.random.choice(indices, self.n_samples, replace=True)
                batch_indices.extend(chosen)
            
            yield batch_indices
            count += 1

    def __len__(self):
        return self.n_batches

# --- 2. MODEL ARCHITECTURE ---
class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=256, alpha=100.0, normalize_input=True):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha 
        self.normalize_input = normalize_input
        self.conv = nn.Conv1d(dim, num_clusters, kernel_size=1, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def forward(self, x):
        N, num_tokens, C = x.shape
        if self.normalize_input: x = F.normalize(x, p=2, dim=2)
        
        x_t = x.permute(0, 2, 1) 
        soft_assign = F.softmax(self.conv(x_t), dim=1) 
        
        x_flatten = x_t.unsqueeze(1)
        centroids = self.centroids.unsqueeze(0).unsqueeze(-1)
        residual = x_flatten - centroids 
        
        vlad = (residual * soft_assign.unsqueeze(2)).sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad

class AnyLocModel(nn.Module):
    def __init__(self, input_dim=1024, pca_dim=256, num_clusters=64):
        super().__init__()
        self.pca = nn.Linear(input_dim, pca_dim)
        self.netvlad = NetVLAD(num_clusters=num_clusters, dim=pca_dim)
        self.compress = nn.Linear(num_clusters * pca_dim, 2048)

    def forward(self, x):
        x = self.pca(x) 
        vlad = self.netvlad(x) 
        output = F.normalize(self.compress(vlad), p=2, dim=1)
        return output

    def init_pca_and_centroids(self, descriptors):
        print(f"Fitting PCA ({self.pca.in_features} -> {self.pca.out_features})...")
        descriptors_np = descriptors.cpu().numpy()
        pca = PCA(n_components=self.pca.out_features)
        pca.fit(descriptors_np)
        self.pca.weight.data.copy_(torch.from_numpy(pca.components_).float())
        self.pca.bias.data.copy_(torch.from_numpy(-pca.mean_ @ pca.components_.T).float())
        
        print(f"Fitting KMeans (K={self.netvlad.num_clusters})...")
        descriptors_proj = pca.transform(descriptors_np)
        kmeans = KMeans(n_clusters=self.netvlad.num_clusters, n_init=10, max_iter=100)
        kmeans.fit(descriptors_proj)
        
        centroids = torch.from_numpy(kmeans.cluster_centers_).float()
        self.netvlad.centroids.data.copy_(centroids)
        self.netvlad.conv.weight.data.copy_((2.0 * self.netvlad.alpha * centroids).unsqueeze(-1))
        self.netvlad.conv.bias.data.copy_(- self.netvlad.alpha * (centroids**2).sum(dim=1))
        print("Initialization Complete.")

# --- 3. DATASET ---
class HotelMetricDataset(Dataset):
    def __init__(self, features_dir, is_train=True):
        self.files = list(glob.glob(os.path.join(features_dir, "**", "*.pt"), recursive=True))
        
        # Filter tiny files
        self.files = [f for f in self.files if os.path.getsize(f) > 1000]
        
        self.classes = sorted(list(set([os.path.basename(os.path.dirname(f)) for f in self.files])))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        if is_train:
            counts = defaultdict(int)
            for f in self.files: counts[os.path.basename(os.path.dirname(f))] += 1
            self.files = [f for f in self.files if counts[os.path.basename(os.path.dirname(f))] >= 2]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.class_to_idx[os.path.basename(os.path.dirname(path))]
        try: features = torch.load(path)
        except: features = torch.zeros(1, 1024)
        
        if features.shape[0] == 0: features = torch.zeros(1, 1024)
        
        # Noise augmentation
        if len(features) > 0:
            noise = torch.randn_like(features) * 0.02 
            features = features + noise

        return features, label

def collate_pad(batch):
    features, labels = zip(*batch)
    max_len = max([f.shape[0] for f in features])
    # Pad to make batch stackable
    padded = [F.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in features]
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)

def batch_hard_triplet_loss(embeddings, labels, margin):
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    labels = labels.unsqueeze(0)
    mask_pos = labels.eq(labels.t())
    mask_neg = ~mask_pos
    
    dist_pos = dist_matrix * mask_pos.float()
    hardest_pos, _ = dist_pos.max(dim=1)
    
    max_dist = dist_matrix.max().detach()
    dist_neg = dist_matrix + max_dist * mask_pos.float()
    hardest_neg, _ = dist_neg.min(dim=1)
    
    loss = torch.clamp(hardest_pos - hardest_neg + margin, min=0.0)
    return loss.mean()

def compute_knn_accuracy(model, train_loader, val_loader):
    model.eval()
    # Gallery
    gallery_embeds, gallery_labels = [], []
    with torch.no_grad():
        for features, labels in train_loader:
            gallery_embeds.append(model(features.to(DEVICE)).cpu())
            gallery_labels.append(labels.cpu())
    gallery_X = torch.cat(gallery_embeds)
    gallery_y = torch.cat(gallery_labels)
    
    # Query
    query_embeds, query_labels = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            query_embeds.append(model(features.to(DEVICE)).cpu())
            query_labels.append(labels.cpu())
    query_X = torch.cat(query_embeds)
    query_y = torch.cat(query_labels)
    
    correct = 0
    chunk_size = 100
    gallery_X_gpu = gallery_X.to(DEVICE)
    gallery_y_gpu = gallery_y.to(DEVICE)
    
    for i in range(0, len(query_X), chunk_size):
        q_batch = query_X[i:i+chunk_size].to(DEVICE)
        y_batch = query_y[i:i+chunk_size].to(DEVICE)
        dists = torch.cdist(q_batch, gallery_X_gpu)
        _, indices = dists.min(dim=1)
        correct += (gallery_y_gpu[indices] == y_batch).sum().item()
        
    return correct / len(query_y)

# --- 4. MAIN ---
if __name__ == "__main__":
    if not os.path.exists(FEATURES_DIR): raise ValueError("Check path!")
    
    print("Loading Dataset...")
    full_dataset = HotelMetricDataset(FEATURES_DIR, is_train=True)
    
    val_count = int(0.1 * len(full_dataset))
    train_count = len(full_dataset) - val_count
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_count, val_count], generator=generator)
    
    # --- FIX IS HERE: PASS THE SUBSET (train_ds) TO THE SAMPLER ---
    batch_sampler = BalancedBatchSampler(train_ds, n_classes=8, n_samples=4)
    
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=collate_pad)
    
    # Standard sequential loaders for validation
    train_gallery_loader = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_pad)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    print("Sampling for Init...")
    sample_features = []
    indices = torch.randperm(len(train_ds))[:200]
    for i in indices:
        feat, _ = train_ds[i]
        if feat.shape[0] > 0: sample_features.append(feat)
    all_patches = torch.cat(sample_features, dim=0)

    model = AnyLocModel(input_dim=INPUT_DIM, pca_dim=PCA_DIM, num_clusters=NUM_CLUSTERS).to(DEVICE)
    model.init_pca_and_centroids(all_patches)

    loss_func = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)

    # 2. Initialize Miner (Finds the hard pairs automatically)
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_acc = 0.0

    print("Starting Training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for features, labels in pbar:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            embeddings = model(features)
            
            # NEW LOSS CALCULATION
            hard_pairs = miner(embeddings, labels)
            loss = loss_func(embeddings, labels, hard_pairs)
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print("Validating...")
        val_acc = compute_knn_accuracy(model, train_gallery_loader, val_loader)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Recall@1: {val_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_anyloc_model.pth")
            print(">>> Saved Best Model!")