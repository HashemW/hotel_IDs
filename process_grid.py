import torch
import cv2
import os
import numpy as np
import gc
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path # Pathlib makes Windows paths way easier

# --- CONFIG (WINDOWS PATHS) ---
# Using raw strings (r"...") prevents slash errors
BASE_DIR = Path(r"D:\Coding\hotels") 
DATASET_ROOT = BASE_DIR / "dataset"
OUTPUT_ROOT = BASE_DIR / "grid_features"

DINO_REPO_DIR = BASE_DIR / "dinov3"
DINO_WEIGHTS_PATH = BASE_DIR / "weights" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DINO_MODEL_NAME = "dinov3_vitl16"

# Image Size (Keep 224 for safety on 8GB VRAM)
IMG_SIZE = 448 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on: {DEVICE}")

# --- MODEL ---
# Force load to GPU
model = torch.hub.load(str(DINO_REPO_DIR), DINO_MODEL_NAME, source='local', weights=str(DINO_WEIGHTS_PATH))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- ADD THIS CONFIG AT THE TOP ---
MASKS_ROOT = BASE_DIR / "dataset" / "train_masks" 

def process_single_image(image_path, output_path):
    if output_path.exists(): return

    try:
        # 1. Load the Original Image
        with Image.open(image_path) as img:
            image = img.convert('RGB')

            # 2. CHECK FOR MASK (Only for training images)
            # Structure usually matches: train_images/123.jpg -> train_masks/123.png
            # We check if a corresponding png exists in the masks folder
            
            # Get relative path (e.g., "00001.jpg")
            # Note: Mask filenames usually match image filenames but might be .png
            mask_name = image_path.stem + ".png" 
            
            # Construct potential mask path. 
            # Note: Sometimes masks are in subfolders matching the train_images structure.
            # We assume a flat structure or matching structure. 
            # Let's try to preserve the parent folder structure if it exists.
            rel_parent = image_path.parent.name # e.g. "hotel_id" folder
            mask_path_guess = MASKS_ROOT / rel_parent / mask_name
            
            # If flat structure in masks folder, try that too
            if not mask_path_guess.exists():
                mask_path_guess = MASKS_ROOT / mask_name

            if mask_path_guess.exists():
                with Image.open(mask_path_guess) as mask:
                    mask = mask.convert('RGBA')
                    mask = mask.resize(image.size, Image.NEAREST)
                    
                    # Create a blank black image to use for occlusion
                    black_block = Image.new("RGB", image.size, (0, 0, 0))
                    
                    # The mask usually has Red/Opaque pixels where the victim is.
                    # We use the mask's Alpha channel to paste Black over the Victim.
                    # This effectively "deletes" the person from the data.
                    image.paste(black_block, (0,0), mask)

            # 3. Transform & Forward Pass (Same as before)
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            if hasattr(model, 'forward_features'):
                output = model.forward_features(input_tensor)
            else:
                output = model(input_tensor)
            
            # Extract features (Same logic as before)
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
                # Remove CLS token
                if features.dim() == 3 and features.shape[1] > 50:
                    features = features[:, 1:, :]
                
                # Save
                features = features.squeeze(0).cpu().float()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(features, output_path)

    except Exception as e:
        print(f"Error {image_path}: {e}")

def main():
    target_folders = ['train_images', 'test_images'] 
    
    gc_step = 0

    for folder in target_folders:
        split_dir = DATASET_ROOT / folder
        if not split_dir.exists(): 
            print(f"Skipping {split_dir} (Not found)")
            continue
        
        print(f"Scanning {folder}...")
        # Pathlib recursive glob is cleaner
        image_files = list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.jpeg")) + list(split_dir.rglob("*.png"))
        
        print(f"Processing {len(image_files)} images...")
        
        for img_path in tqdm(image_files):
            # Calculate output path maintaining structure
            # e.g. C:\hotels\dataset\train\123\img.jpg -> C:\hotels\grid_features\train\123\img.pt
            rel_path = img_path.relative_to(DATASET_ROOT)
            save_path = OUTPUT_ROOT / rel_path.with_suffix(".pt")
            
            process_single_image(img_path, save_path)
            
            gc_step += 1
            if gc_step % 50 == 0:
                gc.collect()

if __name__ == "__main__":
    main()