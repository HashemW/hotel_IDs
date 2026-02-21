import torch
import cv2
import os
import numpy as np
import gc
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

# --- CONFIG ---
BASE_DIR = Path(r"D:\Coding\hotels") 
DATASET_ROOT = BASE_DIR / "dataset"
OUTPUT_ROOT = BASE_DIR / "grid_features_masked" # Save to a NEW folder so we don't overwrite clean ones!
MASKS_ROOT = DATASET_ROOT / "train_masks"

DINO_REPO_DIR = BASE_DIR / "dinov3"
DINO_WEIGHTS_PATH = BASE_DIR / "weights" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DINO_MODEL_NAME = "dinov3_vitl16"

IMG_SIZE = 448 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on: {DEVICE}")

# --- LOAD ALL MASK PATHS ONCE ---
print("Indexing Masks...")
# Find all PNGs in the masks folder (recursive or flat)
all_mask_paths = list(MASKS_ROOT.rglob("*.png"))
if len(all_mask_paths) == 0:
    raise ValueError("No masks found! Check your MASKS_ROOT path.")
print(f"Found {len(all_mask_paths)} occlusion masks.")

# --- MODEL ---
model = torch.hub.load(str(DINO_REPO_DIR), DINO_MODEL_NAME, source='local', weights=str(DINO_WEIGHTS_PATH))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_single_image(image_path, output_path):
    if output_path.exists(): return

    try:
        with Image.open(image_path) as img:
            image = img.convert('RGB')
            
            # --- THE FIX: RANDOM MASK AUGMENTATION ---
            # Pick one random mask from the 5,000 available
            mask_path = random.choice(all_mask_paths)
            
            with Image.open(mask_path) as mask:
                mask = mask.convert('RGBA')
                
                # Resize mask to match the image size (e.g., 448x448)
                # Resampling=NEAREST keeps the sharp red edges (important!)
                mask = mask.resize(image.size, Image.NEAREST)
                
                # Composite the mask onto the image
                # The mask has transparency (Alpha channel), so it only covers the "red" part
                image.paste(mask, (0, 0), mask)
            
            # Now 'image' looks just like the test set (Room + Red Box)
            # ----------------------------------------

            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Forward Pass (Same as before)
        with torch.no_grad():
            if hasattr(model, 'forward_features'):
                output = model.forward_features(input_tensor)
            else:
                output = model(input_tensor)
            
            features = None
            if isinstance(output, dict):
                features = output.get("x_norm_patchtokens", output.get("x_patchtokens"))
            
            if features is None:
                 if isinstance(output, dict):
                    for v in output.values():
                        if isinstance(v, torch.Tensor) and v.dim() == 3:
                            features = v
                            break
                 elif isinstance(output, torch.Tensor):
                    features = output
            
            if features is not None:
                if features.dim() == 3 and features.shape[1] > 50:
                    features = features[:, 1:, :]
                
                features = features.squeeze(0).cpu().float()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(features, output_path)

    except Exception as e:
        print(f"Error {image_path}: {e}")

def main():
    # Only process TRAIN images with masks. 
    # TEST images (in your local folder) are already masked (or dummy), so we treat them normally.
    target_folders = ['train_images'] 
    
    gc_step = 0

    for folder in target_folders:
        split_dir = DATASET_ROOT / folder
        if not split_dir.exists(): continue
        
        print(f"Scanning {folder}...")
        image_files = list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.jpeg")) + list(split_dir.rglob("*.png"))
        
        print(f"Processing {len(image_files)} images...")
        
        for img_path in tqdm(image_files):
            rel_path = img_path.relative_to(DATASET_ROOT)
            save_path = OUTPUT_ROOT / rel_path.with_suffix(".pt")
            
            process_single_image(img_path, save_path)
            
            gc_step += 1
            if gc_step % 50 == 0:
                gc.collect()

if __name__ == "__main__":
    main()