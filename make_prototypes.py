import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import glob
from tqdm import tqdm
from collections import defaultdict
import random
import copy

# --- CONFIG ---
# Update to point to your NEW masked features if you generated them
FEATURES_DIR = r"D:\Coding\hotels\grid_features_masked\train_images" 
MODEL_PATH = "best_anyloc_model_masked.pth"
OUTPUT_VECS = "prototype_vectors.pt"
OUTPUT_IDS = "prototype_labels.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 

# --- MODEL (Must Match Training) ---
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

# --- DATASET ---
class GalleryDataset(Dataset):
    def __init__(self, files):
        self.files = files
        
    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        hotel_id = os.path.basename(os.path.dirname(path))
        try: features = torch.load(path)
        except: features = torch.zeros(1, 1024)
        if features.shape[0] == 0: features = torch.zeros(1, 1024)
        return features, hotel_id

def collate_pad(batch):
    features, ids = zip(*batch)
    max_len = max([f.shape[0] for f in features])
    padded = [F.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in features]
    return torch.stack(padded), ids

# --- METRIC CALCULATOR ---
def calculate_recall_at_5(query_vecs, query_ids, gallery_vecs, gallery_ids):
    # Ensure they are tensors on GPU
    if not torch.is_tensor(query_vecs): query_vecs = torch.stack(query_vecs)
    if not torch.is_tensor(gallery_vecs): gallery_vecs = torch.stack(gallery_vecs)
    
    query_vecs = query_vecs.to(DEVICE)
    gallery_vecs = gallery_vecs.to(DEVICE)
    
    # gallery_ids might be a list of strings, we need to handle that
    # We can't put strings on GPU, so we do index lookup
    
    correct_top5 = 0
    total = len(query_ids)
    
    # Process in chunks to avoid OOM
    chunk_size = 100
    for i in range(0, total, chunk_size):
        q_batch = query_vecs[i:i+chunk_size]
        q_ids_batch = query_ids[i:i+chunk_size]
        
        # Distance
        dists = torch.cdist(q_batch, gallery_vecs)
        
        # Top 5
        _, indices = dists.topk(5, dim=1, largest=False)
        indices = indices.cpu().numpy()
        
        for j, q_id in enumerate(q_ids_batch):
            # Check if correct ID is in the top 5 matches
            retrieved_ids = [gallery_ids[idx] for idx in indices[j]]
            if q_id in retrieved_ids:
                correct_top5 += 1
                
    return correct_top5 / total

def build_prototypes(features, ids):
    """Aggregates raw features into prototypes"""
    grouped = defaultdict(list)
    for i, hid in enumerate(ids):
        grouped[hid].append(features[i])
    
    proto_vecs = []
    proto_ids = []
    
    for hid in sorted(grouped.keys()):
        vecs = torch.stack(grouped[hid])
        # DBA: Mean + Normalize
        proto = F.normalize(vecs.mean(dim=0, keepdim=True), p=2, dim=1)
        proto_vecs.append(proto)
        proto_ids.append(hid)
        
    return torch.cat(proto_vecs), proto_ids

def main():
    print("--- LOADING DATA ---")
    all_files = list(glob.glob(os.path.join(FEATURES_DIR, "**", "*.pt"), recursive=True))
    all_files = [f for f in all_files if os.path.getsize(f) > 1000]
    
    # --- EVALUATION PHASE (90/10 Split) ---
    print("\n--- PHASE 1: EVALUATING IMPROVEMENT ---")
    # Shuffle and split
    random.seed(42)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    model = AnyLocModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # 1. Encode Everything Once
    def encode_dataset(files):
        ds = GalleryDataset(files)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_pad)
        vecs, ids = [], []
        with torch.no_grad():
            for feats, batch_ids in tqdm(loader, desc="Encoding"):
                emb = model(feats.to(DEVICE)).cpu()
                vecs.extend(emb)
                ids.extend(batch_ids)
        return torch.stack(vecs), ids

    print("Encoding Validation Set...")
    val_vecs, val_ids = encode_dataset(val_files)
    
    print("Encoding Train Set (Baseline)...")
    train_vecs, train_ids = encode_dataset(train_files)
    
    # 2. Baseline Score (Val vs Raw Train Images)
    print("Calculating Baseline Accuracy (Standard k-NN)...")
    base_acc = calculate_recall_at_5(val_vecs, val_ids, train_vecs, train_ids)
    print(f"Baseline Recall@5: {base_acc*100:.2f}%")
    
    # 3. Prototype Score (Val vs Clean Prototypes)
    print("Building Prototypes from Train Set...")
    proto_vecs, proto_ids = build_prototypes(train_vecs, train_ids)
    
    print(f"Compressed {len(train_vecs)} images -> {len(proto_vecs)} prototypes.")
    
    print("Calculating Prototype Accuracy (DBA)...")
    proto_acc = calculate_recall_at_5(val_vecs, val_ids, proto_vecs, proto_ids)
    print(f"Prototype Recall@5: {proto_acc*100:.2f}%")
    
    improvement = (proto_acc - base_acc) * 100
    print(f"\n>>> IMPROVEMENT: +{improvement:.2f}% by using Prototypes! <<<\n")

    # --- PRODUCTION PHASE (100% Data) ---
    print("--- PHASE 2: GENERATING FINAL KAGGLE PROTOTYPES ---")
    # We combine Train + Val to use 100% of data for the final file
    print("Merging all data...")
    full_vecs = torch.cat([train_vecs, val_vecs])
    full_ids = train_ids + val_ids
    
    print("Building Final Prototypes...")
    final_proto_vecs, final_proto_ids = build_prototypes(full_vecs, full_ids)
    
    torch.save(final_proto_vecs, OUTPUT_VECS)
    torch.save(final_proto_ids, OUTPUT_IDS)
    print(f"Saved {len(final_proto_vecs)} Final Prototypes to {OUTPUT_VECS}")

if __name__ == "__main__":
    main()