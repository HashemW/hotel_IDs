# 🏨 Hotel-ID: Combatting Human Trafficking with AI

**A State-of-the-Art Metric Learning pipeline to identify hotel rooms from images, aiding investigators in locating victims of human trafficking.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Status](https://img.shields.io/badge/Status-SOTA%20Baseline-brightgreen)

## 📖 Overview

This project was developed for the **Kaggle Hotel-ID to Combat Human Trafficking 2022** competition. The goal is to recognize a specific hotel based on images of guest rooms, even when the images are occluded, blurry, or taken from different angles.

Identifying the hotel where a photograph was taken is a crucial step in human trafficking investigations. This model helps automate that process using advanced **Metric Learning** techniques.

## 🚀 Key Features & Performance

-   **Deep Feature Extraction:** Utilizes **DINOv3 (ViT-Large-16)**, a self-supervised Vision Transformer, processing high-resolution (448x448) images for fine-grained detail.
-   **Learnable Aggregation:** Implements **NetVLAD** (Vector of Locally Aggregated Descriptors) with PCA dimensionality reduction to create robust, distinctive room fingerprints.
-   **Metric Learning:** Trained using **Multi-Similarity Loss** with **Batch Hard Mining** to strictly enforce cluster separation between different hotels.
-   **Victim Masking:** Custom data preprocessing pipeline that occludes victims in training data, forcing the model to learn invariant room features (furniture, wallpaper, layout) rather than human attributes.
-   **Inference Strategy:** Uses **Database Side Feature Augmentation (DBA)** to create noise-free "Hotel Prototypes" for retrieval.

### 📊 Results

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Recall@5** | **77.6%** | Probability that the correct hotel is in the top 5 predictions. |
| **Recall@1** | **65.5%** | Probability that the correct hotel is the #1 prediction. |

*(Scores based on local validation set with victim masking strategy applied)*

## 🛠️ Tech Stack

* **Core:** PyTorch, Torchvision
* **Architecture:** DINOv3 (Backbone), NetVLAD (Aggregator), AnyLoc (Pipeline)
* **Training:** PyTorch Metric Learning (Losses/Miners)
* **Data Processing:** PIL, NumPy, Pandas, Scikit-Learn
* **Hardware:** Optimized for NVIDIA GPUs (CUDA)

## ⚙️ Methodology

1. **Masking & Preprocessing:**
Training images are paired with masks to black out victims. This prevents the model from overfitting to human features (e.g., clothing, faces) and focuses attention on the environment.
2. **Feature Generation (Offline):**
Images are resized to `448x448` and passed through a frozen **DINOv3** backbone. We extract the patch tokens (removing the CLS token) to preserve spatial information, resulting in a `28x28` grid of `1024-d` descriptors.
3. **Aggregation & Training:**
* **PCA:** Reduces descriptors from 1024-d to 256-d.
* **NetVLAD:** Aggregates descriptors into 64 semantic clusters (e.g., "curtains", "carpet").
* **Compression:** A linear layer compresses the final vector to 2048-d.
* The model is trained to minimize the distance between images of the same hotel while maximizing the distance to others.


4. **Prototype Inference:**
Instead of 1:1 image comparison, we average all feature vectors for a specific hotel ID to create a single **Prototype**. Queries are matched against these clean prototypes using Cosine Similarity.

## 📦 Usage

### 1. Install Dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn pytorch-metric-learning tqdm

```

### 2. Extract Features

```bash
python src/process_grid.py

```

*This will scan `dataset/train_images` and save `.pt` tensors to `grid_features/`.*

### 3. Train the Model

```bash
python src/train_masked.py

```

*Trains the NetVLAD/AnyLoc head and saves `best_anyloc_model_masked.pth`.*

### 4. Generate Prototypes

```bash
python src/make_prototypes.py

```

*Creates `prototype_vectors.pt` and `prototype_labels.pt` for fast inference.*

## 🤝 Acknowledgements

* **DINOv3** by Meta Research for the powerful vision backbone.
* **AnyLoc** & **NetVLAD** for the aggregation methodology.
* **Kaggle** for hosting the dataset and competition.

---

*This project is for educational and humanitarian purposes.*
