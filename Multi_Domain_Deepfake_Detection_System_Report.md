# Project Report: Multi-Domain Deepfake Detection System

A comprehensive overview of planning, architecture, step-by-step implementation, and tools used for deepfake detection using spatial-frequency analysis.

---

## 1. Executive Summary & Project Goal
The **Multi-Domain Deepfake Detection System** is designed to identify and classify visual deepfakes (synthetic human faces, face swaps, and facial manipulations) with high accuracy and explainability.

Unlike traditional detectors that rely solely on spatial configurations (analyzing RGB pixel distributions) which can be bypassed by advanced generative models or heavy compression, this system utilizes a **multi-domain spatial-frequency framework**. It integrates spatial textures, global geometry, and frequency-domain anomalies to create a robust detector.

---

## 2. Technical Architecture & Components
The system is divided into four core neural network branches followed by a dynamic fusion module:

```
                  ┌─────────────── BGR Face Crop ──────────────┐
                  │                                            │
                  ▼                                            ▼
┌───────────────────────────────────┐        ┌───────────────────────────────────┐
│     Spatial CNN Branch (RGB)      │        │     Spatial ViT Branch (RGB)      │
│        (EfficientNet-B0)          │        │      (vit_base_patch16_224)       │
│  Captures local textures, edges   │        │ Captures global face geometry,    │
│  and blending boundaries.         │        │ shadows and layout consistency.   │
└─────────────────┬─────────────────┘        └─────────────────┬─────────────────┘
                  │                                            │
                  ▼                                            ▼
         [Local Spatial Embed]                       [Global Spatial Embed]
                  │                                            │
                  └───────────────────────┬────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │   Cross-Attention   │◄───────── [Frequency Embed]
                               │    Fusion Module    │              (FFT & DCT)
                               └──────────┬──────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │  Classification     │
                               │  Logits (Real/Fake) │
                               └─────────────────────┘
```

### 2.1. Spatial CNN Branch (EfficientNet-B0)
* **Goal:** Capture localized artifacts and micro-textures.
* **Details:** EfficientNet-B0 acts as an efficient feature extractor. It captures subtle artifacts such as blending seam inconsistencies around eyes/lips, color mismatch boundaries, and artificial reflections that are localized within specific pixel blocks.

### 2.2. Spatial ViT Branch (Vision Transformer)
* **Goal:** Capture global structural inconsistencies and spatial context.
* **Details:** Vision Transformer (ViT-Base-Patch16-224) splits the face crop into 16x16 non-overlapping patches. By utilizing self-attention, it tracks long-range dependencies across the face to detect global alignment mismatches, irregular shadow casting, and asymmetry across eyes, nose, and chin.

### 2.3. Frequency Domain Branch (FFT & DCT)
* **Goal:** Capture frequency anomalies left by upsampling layers in generative models.
* **Details:** Generative networks (GANs/Diffusion) generate images by upsampling noise, which inevitably leaves periodic grid-like pattern anomalies in the frequency spectrum. The branch computes:
  1. **2D Fast Fourier Transform (FFT):** To identify periodic patterns.
  2. **2D Discrete Cosine Transform (DCT):** To isolate high-frequency noise deviations.
* These transforms are stacked into a 2-channel tensor and processed by a dedicated shallow CNN.

### 2.4. Cross-Attention Fusion Module
* **Goal:** Learn context-dependent feature weighting instead of simple concatenation.
* **Details:** Features from the spatial branches (CNN, ViT) and the frequency branch are projected into a shared feature space. A Multi-Head Cross-Attention mechanism dynamically weights the features based on inputs. For instance, if an image is highly compressed (blurring spatial textures), the module automatically shifts its attention weight toward the frequency branch to maintain robust detection.

---

## 3. Step-by-Step Implementation Process

### Step 1: Real-World Dataset Selection (FaceForensics++)
Originally, the pipeline was configured to load Celeb-DF. It was transitioned to **FaceForensics++ (FF++)**, a widely accepted research benchmark dataset.
* We integrated streaming from the Hugging Face dataset `TsienDragon/ffplusplus_c23_frames` (which contains frame-level crops from the C23 compression level).
* The script streams 100 balanced face samples (50 Real, 50 Fake) to train locally.

### Step 2: Facial Detection and Preprocessing Pipeline
* For raw video/image inputs, system runs frame extraction using OpenCV.
* The face detector (MTCNN) detects facial bounding boxes. The face is cropped, padded by 10% to capture boundary blending details, and resized to **224x224**.
* Log-magnitude FFT and DCT maps are generated from the grayscale version of the face crop to serve as the frequency branch inputs.
* The dataset is split at the **Video Level** (70% Train, 14% Val, 16% Test) to ensure zero frame leakage between training and evaluation splits.

### Step 3: Model Training Configuration
* The training script initializes pretrained EfficientNet-B0 and ViT weights from ImageNet.
* Training is configured with:
  * **Optimizer:** AdamW (with a learning rate of `1e-4` and weight decay of `1e-5`).
  * **Loss Function:** CrossEntropyLoss.
  * **Learning Rate Scheduler:** CosineAnnealingLR.
* The model trained successfully for 3 epochs:
  * **Epoch 1:** Train Loss: 0.3113 | Val Accuracy: 100.00%
  * **Epoch 2:** Train Loss: 0.1889 | Val Accuracy: 100.00%
  * **Epoch 3:** Train Loss: 0.0978 | Val Accuracy: 100.00%
* The trained model checkpoint is saved to `outputs/checkpoints/best_model.pth`.

### Step 4: Explainability (Grad-CAM Activation Map)
To provide explainability for decisions, a Grad-CAM hook intercepts gradients flowing to the final convolutional layer of the spatial CNN branch. It highlights which facial pixels contributed most to the classification (e.g., eyes, lips, or blending edges) and overlays this heatmap onto the image for the end-user.

### Step 5: Web Application Serving
* **Flask Backend (`app.py`):** Serves REST API endpoints on port `5000` to process uploaded images or video files, run frame extraction/pre-processing, perform model forward inference, generate Grad-CAM, and return prediction probabilities.
* **React Frontend (`App.jsx`):** Provides a visual dashboard running on port `5173`. Users can drag and drop media files, visualize prediction meters, and review Grad-CAM heatmaps.

---

## 4. Tools & Technologies Used
To build this project, we leveraged a modern machine learning and web development stack:

* **PyTorch & Torchvision:** Core framework for defining layers, running the training loop, backward gradient computation, and tensor manipulation.
* **Timm (Pytorch Image Models):** Used to load pre-trained deep architectures (EfficientNet, ViT) and customize their classification heads.
* **OpenCV (cv2):** Used for reading images, video streams, resizing, and rendering the Grad-CAM overlays.
* **MTCNN (Multi-task Cascaded Convolutional Networks):** Robust face detection library used during preprocessing.
* **Flask & Flask-CORS:** Python web framework to expose REST APIs to the frontend.
* **Vite & React:** High-performance frontend bundler and UI library for a responsive dashboard.
* **Matplotlib:** To construct the side-by-side plots for the analyzed face crop and the Grad-CAM heatmap.

---

## 5. Interviewer Q&A Prep

### Q: Why did you combine CNN and ViT branches?
* **Answer:** CNNs are naturally biased toward local features due to the sliding window kernel (translational invariance). They are great at detecting micro-textures and local blending borders. Vision Transformers (ViT) process the image as patches and calculate global attention, capturing structural inconsistencies across the entire face (like uneven lighting or eye-to-chin ratio). Combining both ensures we capture both local and global artifacts.

### Q: How does the frequency branch detect deepfakes?
* **Answer:** Generative networks synthesize images using layers like transpose convolutions or bilinear upsampling. This process creates high-frequency grid-like artifacts (known as checkerboard artifacts) that are often imperceptible to the human eye or standard CNNs. By converting the crop to 2D FFT and DCT log-magnitude spectrums, these periodic anomalies manifest as bright grid lines or spikes, which a classifier can easily identify as artificial.

### Q: Why does the system classify non-facial images (like buildings) as REAL?
* **Answer:** The system is an "in-domain" classifier trained strictly on face crops. When an image with no face is uploaded, the face detector falls back to a center-crop. Because the crop does not contain any of the synthetic face signatures or checkerboard anomalies the model has learned, the output defaults to REAL. In a production pipeline, an validation checker should be added before the model to reject images that do not contain a human face.

### Q: What is the benefit of the Cross-Attention Fusion module?
* **Answer:** Simple concatenation (stacking vectors) assumes all features are equally important. Cross-attention computes interactive query-key weights, letting features from one branch guide the features of another. For example, if spatial features are corrupted by high image compression, the cross-attention layers automatically weight the frequency features higher, resulting in better generalization.
