# Multi-Domain Deepfake Detection System

A unified, multi-modal, and multi-domain deep learning system designed to detect synthetic faces and visual manipulations in video or image inputs. The system processes inputs across **three parallel neural branches** (Spatial, Semantic, and Spectral Frequency) and dynamically fuses their signatures using a **Multi-Head Self-Attention Fusion Module** to render predictions along with **Grad-CAM explainability heatmaps**.

---

## 🚀 Quick Start: Running the Web Application

The system features a **Flask backend** REST API and a **React (Vite) frontend** client.

### 1. Start the Flask Backend Server
Run from the root of the repository:
```powershell
python app.py
```
- **Port**: `5000` (`http://localhost:5000`)
- **Key Feature**: Loads the PyTorch model checkpoint globally at startup for instant upload evaluations.
- **REST API Code**: [app.py](file:///d:/Multi-Domain%20Deepfake%20Detection%20System/app.py)

### 2. Start the React Frontend Dashboard
In a separate terminal window, navigate to the `frontend/` directory and start the Vite dev server:
```powershell
cd frontend
npm run dev
```
- **Port**: `5173` (`http://localhost:5173`)
- **Web App Code**: [App.jsx](file:///d:/Multi-Domain%20Deepfake%20Detection%20System/frontend/src/App.jsx) | Styles: [index.css](file:///d:/Multi-Domain%20Deepfake%20Detection%20System/frontend/src/index.css)

Open `http://localhost:5173` in any browser to upload video clips or face image files, trigger predictions, and view the heatmaps.

---

## 📂 Project Architecture

```
d:/Multi-Domain Deepfake Detection System/
├── config.yaml                    # Main hyperparameters, network config, and folder paths
├── requirements.txt                # Python backend dependencies
├── app.py                          # Flask REST API backend server
├── predict.py                      # CLI inference tool for video or face image files
├── verify_pipeline.py              # Automatic validation runner for the entire pipeline
├── src/                            # Backend source directories
│   ├── preprocessing/              # Preprocessing algorithms
│   │   ├── extract_frames.py      # Face cropping (MTCNN) from video files
│   │   └── preprocess.py          # Coordinate extraction and dataset splitting
│   ├── models/                    # Model structures
│   │   ├── cnn_branch.py          # Spatial ConvNet (EfficientNet-B0)
│   │   ├── vit_branch.py          # Semantic Transformer (ViT-B/16)
│   │   ├── frequency_branch.py    # Spectral anomaly classifier (FFT/DCT inputs)
│   │   ├── fusion_module.py       # Self-Attention projections & fusion layers
│   │   └── detector.py            # Unified PyTorch detector orchestrator
│   ├── utils/                      # Helper scripts
│   │   ├── dataset.py             # Custom datasets and transforms
│   │   └── generate_synthetic_data.py # Mock data creator for validation runs
│   ├── training/                  # Model training logic
│   │   └── train.py               # Optimizers, schedules, and checkpoint exports
│   └── evaluation/                # Performance analytics
│       ├── evaluate.py            # Computes ROC, Accuracy, F1, EER, and plots
│       └── gradcam.py             # Grad-CAM heatmap localization generator
└── frontend/                       # React Web UI
    ├── index.html                  # Main layout with customized SEO titles
    ├── src/
    │   ├── main.jsx                # React app entry point
    │   ├── App.jsx                 # Dashboard views, upload handlers, and state management
    │   └── index.css               # Dark-mode glassmorphic styling system
```

---

## ⚙️ Hyperparameter Configuration
All paths, neural dimensions, and preprocessing steps are configured in [config.yaml](file:///d:/Multi-Domain%20Deepfake%20Detection%20System/config.yaml):
- **Paths**: Defines raw file input sources, model checkpoints, and evaluation export directories.
- **Preprocessing**: Configures the face resolution (`224x224`) and FFT map settings.
- **Model**: Details feature dimensions for visual branches and dimensions of the self-attention fusion block.

---

## 🛠️ Step-by-Step Pipeline Execution Manual

You can run individual blocks of the deep learning pipeline from the terminal using the commands below:

### Phase 1: Environment Setup
Ensure all backend dependencies are installed:
```powershell
pip install -r requirements.txt
```

### Phase 2: Generating Verification Datasets
To verify the system without downloading massive public deepfake datasets, generate synthetic video samples:
```powershell
python src/utils/generate_synthetic_data.py
```
- **Action**: Generates synthetic videos with alternating textures under `data/raw_videos/`.
- **Script**: [generate_synthetic_data.py](file:///d:/Multi-Domain%20Deepfake%20Detection%20System/src/utils/generate_synthetic_data.py)

### Phase 3: Run Preprocessing
Extract faces and compute frequency domain representations:
```powershell
python src/preprocessing/preprocess.py
```
- **Action**: Runs MTCNN face detection on extracted frames, computes 2D FFT/DCT spectral maps, and splits the data into train/val/test partitions listed inside `data/metadata.csv`.
- **Script**: [preprocess.py](file:///d:/Multi-Domain%20Deepfake%20Detection%20System/src/preprocessing/preprocess.py)

### Phase 4: Model Training
Train the multi-branch network:
```powershell
python src/training/train.py
```
- **Action**: Runs backpropagation across the branches, optimizing weights using AdamW. Checks validation losses and saves the top state weights to `outputs/checkpoints/best_model.pth`.
- **Script**: [train.py](file:///d:/Multi-Domain%20Deepfake%20Detection%20System/src/training/train.py)

### Phase 5: Metric Evaluation
Evaluate model performance against the test split:
```powershell
python src/evaluation/evaluate.py
```
- **Action**: Generates a metrics scorecard (Accuracy, F1, AUC, EER) in JSON format and saves a Confusion Matrix plot to `outputs/evaluation/confusion_matrix.png`.
- **Script**: [evaluate.py](file:///d:/Multi-Domain%20Deepfake%20Detection%20System/src/evaluation/evaluate.py)

### Phase 6: CLI Single-Sample Predictions
Run predictions on custom local files:
```powershell
# Scenario A: Analyze a video file
python predict.py --video "data/raw_videos/fake/fake_1.mp4"

# Scenario B: Analyze a face image
python predict.py --image "data/preprocessed/faces/real_1_frame_0.jpg"
```
- **Action**: Runs preprocessing on the fly, computes model logits, and generates a side-by-side Grad-CAM heatmap showing attention localization.
- **Script**: [predict.py](file:///d:/Multi-Domain%20Deepfake%20Detection%20System/predict.py)

---

## 🔍 Automation: End-to-End Verification Pipeline
You can run the entire deep learning cycle—generating synthetic datasets, preprocessing, training for 1 epoch, running validations, evaluations, and exporting Grad-CAM overlays—in a single command:
```powershell
python verify_pipeline.py
```
- **Script**: [verify_pipeline.py](file:///d:/Multi-Domain%20Deepfake%20Detection%20System/verify_pipeline.py)
