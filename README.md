# 🐼🎨 CUT-GAN Panda Stylization Pipeline  
### YOLOv5 detection → panda cropping → CUT-GAN style transfer → reinsertion into original photos

This project implements a complete computer vision and generative modeling pipeline that:

1. **Detects pandas** in raw photos using YOLOv5  
2. **Crops pandas automatically** using detected bounding boxes  
3. **Trains a CUT-GAN** to stylize pandas using an unpaired target style dataset (e.g., Van Gogh)  
4. **Stylizes panda crops** using the trained generator  
5. **Reinserts stylized pandas** back into the original photos  
6. **Displays full before/after comparisons**

All code is contained in the Jupyter notebook: **`mil4.ipynb`**

---

## 📌 Key Features
- 🐼 YOLOv5 panda detection  
- ✂️ Automatic image cropping  
- 🎨 CUT-GAN generator + PatchGAN discriminator implemented from scratch  
- 🔧 Epoch-by-epoch checkpointing  
- 🖼 Visualizations of crops, stylized outputs, and final merged results  
- ⚡ Apple Silicon MPS acceleration supported  

---

## 📂 Project Structure

```
project/
│
├── mil4.ipynb
│
├── yolov5/yolo_results/panda_boxes3/
│   ├── *.JPG                # Original panda images
│   └── labels/*.txt         # YOLO bounding boxes
│
├── cropped_pandas/          # Auto-generated panda crops
├── gan_panda/vg_dataset/    # Style dataset (e.g., Van Gogh paintings)
├── gan_panda/checkpoints_cut/
├── styled_pandas_vg/        # Stylized GAN outputs
└── final_images/            # Stylized pandas merged back into original photos
```

---

## 🛠 Installation

```bash
pip install torch torchvision pillow matplotlib tqdm glob2
```

(Optional)

```bash
pip freeze > requirements.txt
```

---

# 🚀 Full Pipeline Overview

---

## 1. Import + Path Setup

All project paths are configured automatically via `pathlib`.  
The device is selected via:

```python
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

---

## 2. YOLOv5 Detection Demo

A YOLOv5 prediction image is loaded and displayed with bounding boxes.

---

## 3. Cropping Pandas From YOLOv5 Bounding Boxes

YOLO `.txt` label files are parsed and converted into pixel coordinates.  
Each panda is cropped and saved to `cropped_pandas/`.

```
Total original images: 220
Cropped pandas: 202
```

---

## 4. CUT-GAN Dataset + Architecture

The notebook includes implementations of:

- `ImageDataset` (with augmentations)  
- `ResnetBlock`  
- `Generator` (ResNet-based CUT generator)  
- `PatchDiscriminator`  

Training hyperparameters:

```
IMG_SIZE = 256
BATCH_SIZE = 1
LR = 2e-4
EPOCHS = 40
```

---

## 5. Training the GAN

Training loop includes:

- Discriminator update  
- Generator adversarial + L1 loss update  
- Checkpoint saved each epoch

```
Epoch 1/40
Saved checkpoint: .../G_epoch_001.pth
...
```

---

## 6. Stylizing the Panda Crops

After training completes, the latest checkpoint (e.g., `G_epoch_040.pth`) is loaded.  
Each cropped panda is passed through the generator and saved to:

```
styled_pandas_vg/
```

The notebook stores the first two `(original, stylized)` pairs for visualization.

---

## 7. Before and After Stylization (Side-by-Side)

The notebook displays:

```
Original Crop #1     |    Van Gogh Style #1
Original Crop #2     |    Van Gogh Style #2
```

---

## 8. Reinserting Stylized Pandas Into Original Images

Each stylized panda crop is resized back to bounding-box size and pasted into the original image at `(x1, y1)`.

Output images are saved to:

```
final_images/{image_name}_final.jpg
```

```
Saved: final_images/IMG_2085_final.jpg
Stylized pandas merged back into original images.
```

---

## 9. Final Photo Comparison

The notebook displays a final before/after:

```
Original Image    |    Final Stylized Panda Image
```

---

# ▶️ Running the Notebook

```bash
jupyter notebook mil4.ipynb
```

---

# 💡 Tips for Better Results

- Use a high-quality style dataset  
- Train for 60–100 epochs for smoother textures  
- Ensure YOLOv5 bounding boxes are accurate  
- Experiment with different art styles  
