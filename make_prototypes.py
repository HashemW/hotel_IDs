import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import glob
from tqdm import tqdm
from collections import defaultdict

# --- CONFIG ---
FEATURES_DIR = r"D:\Coding\hotels\grid_features\train_images"
MODEL_PATH = "best_anyloc_model_masked.pth"
OUTPUT_VECS = "prototype_vectors.pt"
OUTPUT_IDS = "prototype_labels.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 # Faster inference

# --- SAME MODEL ARCHITECTURE ---
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

# --- DATASET THAT RETURNS HOTEL ID STRINGS ---
class GalleryDataset(Dataset):
    def __init__(self, features_dir):
        self.files = list(glob.glob(os.path.join(features_dir, "**", "*.pt"), recursive=True))
        # Filter small files
        self.files = [f for f in self.files if os.path.getsize(f) > 1000]
        
    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        # Extract the REAL folder name (Hotel ID)
        # e.g. .../train_images/12345/img.pt -> "12345"
        hotel_id = os.path.basename(os.path.dirname(path))
        
        try: features = torch.load(path)
        except: features = torch.zeros(1, 1024)
        if features.shape[0] == 0: features = torch.zeros(1, 1024)
        
        return features, hotel_id

def collate_pad(batch):
    features, ids = zip(*batch)
    max_len = max([f.shape[0] for f in features])
    padded = [F.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in features]
    return torch.stack(padded), ids # ids is a tuple of strings

def main():
    print("--- GENERATING PROTOTYPES ---")
    
    # 1. Load Model
    model = AnyLocModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # 2. Dataset
    ds = GalleryDataset(FEATURES_DIR)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_pad)
    
    print(f"Processing {len(ds)} images...")
    
    # 3. Aggregate Vectors by Hotel ID
    hotel_vectors = defaultdict(list)
    
    with torch.no_grad():
        for features, ids in tqdm(loader):
            # Encode
            embeddings = model(features.to(DEVICE)).cpu()
            
            # Group by ID
            for i, hotel_id in enumerate(ids):
                hotel_vectors[hotel_id].append(embeddings[i])
    
    # 4. Compute Averages (Prototypes)
    print("Averaging vectors...")
    final_vectors = []
    final_ids = []
    
    # Sort IDs so our index matches alphabetical order (Safety first!)
    sorted_ids = sorted(hotel_vectors.keys())
    
    for hid in tqdm(sorted_ids):
        # Stack all vectors for this hotel
        vecs = torch.stack(hotel_vectors[hid])
        
        # Average & Normalize (DBA)
        # We re-normalize so the length is 1.0 (Unit Sphere)
        prototype = F.normalize(vecs.mean(dim=0, keepdim=True), p=2, dim=1)
        
        final_vectors.append(prototype)
        final_ids.append(hid)
        
    # Stack into one big tensor [Num_Hotels, 2048]
    prototype_tensor = torch.cat(final_vectors)
    
    print(f"Created {len(final_ids)} prototypes.")
    print(f"Shape: {prototype_tensor.shape}")
    
    # 5. Save
    torch.save(prototype_tensor, OUTPUT_VECS)
    torch.save(final_ids, OUTPUT_IDS)
    print("Saved successfully!")

if __name__ == "__main__":
    main()