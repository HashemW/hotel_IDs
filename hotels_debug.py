import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- CONFIG (Matching your real project) ---
INPUT_DIM = 1024       # DinoV2 Vit-L features
PCA_DIM = 256          # Your PCA target
NUM_CLUSTERS = 64      # Your NetVLAD clusters
IMG_SIZE = 448         # 448x448 image
PATCH_SIZE = 14        # Dinov2 usually uses 14 (giving 32x32 grid), but if you used 16 it's 28x28. 
                       # Let's assume 14 for standard DinoV2-L/14, or 16 for L/16.
                       # If 448 / 14 = 32. 32*32 = 1024 patches.
                       # If 448 / 16 = 28. 28*28 = 784 patches.
                       # Let's use 784 (28x28) as you mentioned.
NUM_PATCHES = 784      

def pause(msg=""):
    print(f"\n{msg}")
    input(">>> Press ENTER to execute next step...")
    print("-" * 60)

print("############################################################")
print("#  INTERACTIVE HOTEL CODE WALKTHROUGH                      #")
print("#  Dimensions: 1024 -> 256 (PCA) -> 64 Clusters (NetVLAD)  #")
print("############################################################")

# ==============================================================================
# STEP 1: LOADING DATA
# ==============================================================================
pause("STEP 1: SIMULATING DATA LOADING")

print("LINE OF CODE: features = torch.load(path)")
print("EXPLANATION:  We are loading the .pt file you saved earlier.")
print("              You mentioned specific tensor sizes. Let's create a fake one")
print("              representing ONE image processed by DinoV2.")

# Create dummy data
# Shape: (Number of Patches, Feature Dimension)
image_features = torch.randn(NUM_PATCHES, INPUT_DIM)

print(f"\n[BEFORE] File on disk...")
print(f"[AFTER]  Loaded Tensor Shape: {image_features.shape}")
print("         (784 rows = 784 squares in the grid)")
print("         (1024 cols = 1024 numbers describing each square)")

# ==============================================================================
# STEP 2: BATCHING (The Collate Function)
# ==============================================================================
pause("STEP 2: CREATING A BATCH (The Collate Function)")

print("LINE OF CODE: return torch.stack(padded)")
print("EXPLANATION:  The GPU works faster if we send multiple images at once.")
print("              We need to add a 'Batch Dimension' to the front.")
print("              If we had 2 images, we would stack them.")

# Create a batch of 2 identical images for demonstration
batch = torch.stack([image_features, image_features])

print(f"\n[BEFORE] Single Tensor: {image_features.shape}")
print(f"[AFTER]  Batch Tensor:  {batch.shape}")
print("         (Batch_Size, Patches, Input_Dim)")

# ==============================================================================
# STEP 3: PCA LAYER
# ==============================================================================
pause("STEP 3: DIMENSION REDUCTION (PCA)")

print("LINE OF CODE: self.pca = nn.Linear(input_dim, pca_dim)")
print("              x = self.pca(x)")
print("EXPLANATION:  1024 numbers per patch is too much math for NetVLAD.")
print("              We use a Linear layer (PCA) to shrink the information.")
print("              We are compressing 1024 characteristics down to 256.")

pca_layer = nn.Linear(INPUT_DIM, PCA_DIM)
# Fake weights for demo
with torch.no_grad():
    x_pca = pca_layer(batch)

print(f"\n[BEFORE] Shape: {batch.shape}  <-- Too fat (1024)")
print(f"[AFTER]  Shape: {x_pca.shape}   <-- Just right (256)")
print("         Note: The number of patches (784) did NOT change. Just the depth.")

# ==============================================================================
# STEP 4: NETVLAD - NORMALIZATION
# ==============================================================================
pause("STEP 4: NETVLAD START - PRE-NORMALIZATION")

print("LINE OF CODE: x = F.normalize(x, p=2, dim=2)")
print("EXPLANATION:  Before comparing patches, we must 'normalize' them.")
print("              Imagine one patch is very bright (values like 100, 200) and")
print("              another is dark (values like 0.1, 0.2).")
print("              Normalization makes them comparable by scaling length to 1.")

x_norm = F.normalize(x_pca, p=2, dim=2)
# Let's prove it works by checking the length of the first vector
vector_length = torch.norm(x_norm[0, 0, :]).item()

print(f"\n[RESULT] Vector length of first patch: {vector_length:.4f}")
print("         (Should be close to 1.0)")

# ==============================================================================
# STEP 5: NETVLAD - ASSIGNMENT (The Conv1d Trick)
# ==============================================================================
pause("STEP 5: CLUSTER ASSIGNMENT (The Conv1d Trick)")

print("LINE OF CODE: x_t = x.permute(0, 2, 1)")
print("              assignment_scores = self.conv(x_t)")
print("EXPLANATION:  We have 64 'Cluster Centers' (visual words).")
print("              We need to match every patch against every center.")
print("              The code uses a Conv1d layer as a fast trick to do this dot-product.")

# 1. Permute
x_t = x_norm.permute(0, 2, 1)
print(f"\n[ACTION] Permuting dimensions for Conv1d...")
print(f"         Old: {x_norm.shape} (Batch, Patches, Dim)")
print(f"         New: {x_t.shape} (Batch, Dim, Patches)")

# 2. Convolution
conv = nn.Conv1d(PCA_DIM, NUM_CLUSTERS, kernel_size=1, bias=True)
scores = conv(x_t)

print(f"\n[ACTION] Running Conv1d (Score Calculation)...")
print(f"         Output Shape: {scores.shape} (Batch, Clusters, Patches)")
print("         Meaning: For each of the 784 patches, we have 64 scores.")
print("         (High score = Patch looks like this Cluster)")

# ==============================================================================
# STEP 6: NETVLAD - SOFTMAX
# ==============================================================================
pause("STEP 6: CALCULATING PROBABILITIES (Softmax)")

print("LINE OF CODE: soft_assign = F.softmax(scores, dim=1)")
print("EXPLANATION:  The raw scores from the Conv layer are confusing numbers.")
print("              Softmax turns them into percentages.")
print("              e.g., 'This patch is 80% Cluster A, 10% Cluster B...'")

soft_assign = F.softmax(scores, dim=1)

print(f"\n[RESULT] Shape is still: {soft_assign.shape}")
print("         Let's look at the first patch's assignment:")
print(f"         Sum of probabilities: {torch.sum(soft_assign[0, :, 0]).item():.2f}")
print("         (Should be 1.0, meaning 100%)")

# ==============================================================================
# STEP 7: NETVLAD - RESIDUALS (The Hardest Part)
# ==============================================================================
pause("STEP 7: CALCULATING RESIDUALS (The Core Math)")

print("LINE OF CODE: residual = x_flatten - centroids")
print("EXPLANATION:  This is the magic of VLAD (Vector of Locally Aggregated Descriptors).")
print("              We don't just say 'Patch X belongs to Cluster Y'.")
print("              We calculate: (Patch X) MINUS (Cluster Y Center).")
print("              This records *how* the patch differs from the center.")

# Setup dimensions for subtraction
centroids = nn.Parameter(torch.rand(NUM_CLUSTERS, PCA_DIM))

# (Batch, 1, Dim, Patches)
x_flatten = x_t.unsqueeze(1)
# (1, Clusters, Dim, 1)
c_expand = centroids.unsqueeze(0).unsqueeze(-1)

print(f"\n[DEBUG]  Reshaping for subtraction...")
print(f"         Features:  {x_flatten.shape}")
print(f"         Centroids: {c_expand.shape}")

residual = x_flatten - c_expand

print(f"\n[RESULT] Residual Shape: {residual.shape}")
print("         (Batch, Clusters, Dim, Patches)")
print("         We now have a difference vector for EVERY patch against EVERY cluster.")

# ==============================================================================
# STEP 8: NETVLAD - POOLING
# ==============================================================================
pause("STEP 8: POOLING (Summing it up)")

print("LINE OF CODE: vlad = (residual * soft_assign).sum(dim=-1)")
print("EXPLANATION:  We multiply the residuals by the assignment weight.")
print("              (If a patch doesn't belong to Cluster A, we ignore its residual).")
print("              Then we SUM all the residuals for that cluster.")
print("              This collapses the 'Patches' dimension.")

# Fix dimensions for multiplication
soft_assign_expanded = soft_assign.unsqueeze(2) # (Batch, Clusters, 1, Patches)
weighted = residual * soft_assign_expanded
vlad_vector = weighted.sum(dim=-1) # Sum over patches

print(f"\n[RESULT] VLAD Vector Shape: {vlad_vector.shape}")
print("         (Batch, Clusters, Dim)")
print(f"         We have eliminated the 784 patches. Now we just have {NUM_CLUSTERS} cluster summaries.")

# ==============================================================================
# STEP 9: FINAL FLATTENING
# ==============================================================================
pause("STEP 9: FLATTENING")

print("LINE OF CODE: vlad = vlad.view(x.size(0), -1)")
print("EXPLANATION:  We have a matrix of (64 clusters x 256 dimensions).")
print("              We simply unroll this into one gigantic long line.")

final_vlad = vlad_vector.view(batch.size(0), -1)

print(f"\n[BEFORE] {vlad_vector.shape} (Batch, 64, 256)")
print(f"[AFTER]  {final_vlad.shape} (Batch, 16384)")
print("         (Because 64 * 256 = 16384)")

# ==============================================================================
# STEP 10: COMPRESSION (The Output)
# ==============================================================================
pause("STEP 10: COMPRESSION")

print("LINE OF CODE: output = self.compress(vlad)")
print("EXPLANATION:  16,384 is too big for a search engine/database.")
print("              We use one final Linear layer to shrink it to 2048.")

compressor = nn.Linear(NUM_CLUSTERS * PCA_DIM, 2048)
final_output = compressor(final_vlad)

print(f"\n[FINAL RESULT] {final_output.shape}")
print("               This is the unique 'Fingerprint' for the hotel room image.")
print("\nDONE.")