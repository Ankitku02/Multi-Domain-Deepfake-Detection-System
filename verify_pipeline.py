import os
import sys

# Add project root to sys.path to ensure packages can be imported cleanly
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import module functions
from src.utils import generate_synthetic_data
from src.preprocessing import preprocess
from src.training import train
from src.evaluation import evaluate, gradcam

def main():
    print("==================================================================")
    print("   MULTI-DOMAIN DEEPFAKE DETECTION SYSTEM VERIFICATION PIPELINE   ")
    print("==================================================================")
    
    # Step 1: Synthetic Data Generation
    print("\n[Step 1/5] Generating Synthetic Videos...")
    generate_synthetic_data.main()
    
    # Step 2: Preprocessing
    print("\n[Step 2/5] Running Preprocessing (MTCNN frame/face crops, FFT/DCT maps)...")
    preprocess.main()
    
    # Step 3: Train for 1 epoch to verify backward pass & optimizer
    print("\n[Step 3/5] Training Model (1 Epoch for Pipeline Verification)...")
    success = train.main(epochs_override=1)
    if not success:
        print("\nError: Training pipeline failed. Aborting verification.")
        sys.exit(1)
        
    # Step 4: Evaluate Model
    print("\n[Step 4/5] Running Evaluation Metrics on Test Set...")
    evaluate.main()
    
    # Step 5: Grad-CAM Explainability
    print("\n[Step 5/5] Generating Explainability Grad-CAM Heatmap...")
    gradcam.main()
    
    print("\n==================================================================")
    print("   VERIFICATION SUCCESSFUL: ALL PIPELINE STAGES COMPLETED OK!   ")
    print("==================================================================")

if __name__ == "__main__":
    main()
