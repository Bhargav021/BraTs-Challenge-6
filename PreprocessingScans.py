#declaring:
processed_patients = set()
skipped_patients = set()
errored_patients = {}

import os
import glob
import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm
import cv2
import traceback
import subprocess
import sys
import tempfile
# --- Import torchio and the local, self-contained NyulStandardization class ---
import torchio as tio
from nyul_standardization import NyulStandardization

# =====================================================================================
# Configuration
# =====================================================================================
# --- Use raw strings for Windows path compatibility ---
data_root = r"BraTS-PEDs2024_Training\BraTS-PEDs2024_Training"
modalities = ["t1c", "t1n", "t2f", "t2w"]
save_dir = r"G:\My Drive\InputScans_Final"
os.makedirs(save_dir, exist_ok=True)

# --- Configuration for Resampling and Standardization ---
# TARGET_SPACING will be determined dynamically
TARGET_ORIENTATION = "RAI" # A standard anatomical orientation
NYUL_LANDMARKS_PATH = os.path.join(save_dir, "nyul_landmarks.npy")
NYUL_TRAINING_PATIENT_COUNT = 20 # Use first 20 patients to train the normalizer

# =====================================================================================
# Helper Functions (Rewritten for Stability)
# =====================================================================================

def get_median_spacing(patient_paths):
    """Analyzes the dataset to find the median voxel spacing using nibabel."""
    spacings = []
    for patient_path in tqdm(patient_paths, desc="Analyzing Voxel Spacings"):
        patient_id = os.path.basename(patient_path)
        t1c_path = os.path.join(patient_path, f"{patient_id}-t1c.nii.gz")
        if os.path.exists(t1c_path):
            try:
                img_nib = nib.load(t1c_path)
                spacings.append(img_nib.header.get_zooms()[:3])
            except Exception as e:
                print(f"Could not read spacing for {patient_id}: {e}")
    
    if not spacings:
        raise RuntimeError("Could not read any spacings from the dataset. Check paths.")
        
    spacings_np = np.array(spacings)
    median_spacing = np.median(spacings_np, axis=0)
    return tuple(median_spacing)

def resample_and_reorient(nib_image, target_spacing, is_label=False):
    """Resamples and reorients a nibabel image using scipy and nibabel."""
    canonical_img = nib.as_closest_canonical(nib_image)
    data = canonical_img.get_fdata()
    original_spacing = canonical_img.header.get_zooms()[:3]
    resample_factor = [orig / target for orig, target in zip(original_spacing, target_spacing)]
    order = 0 if is_label else 3
    resampled_data = zoom(data, resample_factor, order=order, mode='nearest')
    return resampled_data

def run_sitk_features(input_tensor, temp_dir, patient_id):
    """
    Runs the unstable SimpleITK features in an isolated subprocess to prevent crashes.
    """
    print(f"[spoof.py] Starting SimpleITK features for {patient_id}")
    
    # Use save_dir for intermediate files
    input_path = os.path.join(save_dir, f'{patient_id}_sitk_input.pt')
    output_path = os.path.join(save_dir, f'{patient_id}_sitk_output.pt')
    log_path = os.path.join(save_dir, 'sitk_helper_error.log')

    # Save the input tensor for the helper script
    try:
        torch.save(input_tensor, input_path)
        print(f"[spoof.py] Saved input tensor to {input_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save input tensor: {e}")

    # Provide the absolute path to the helper script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    helper_script_path = os.path.join(script_dir, 'run_sitk.py')
    
    if not os.path.exists(helper_script_path):
        raise RuntimeError(f"Helper script not found at: {helper_script_path}")
    
    command = [sys.executable, helper_script_path, input_path, output_path]
    print(f"[spoof.py] Running command: {' '.join(command)}")

    # Run the subprocess with more verbose output capture
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 minute timeout
            cwd=script_dir  # Set working directory
        )
        print(f"[spoof.py] Subprocess return code: {result.returncode}")
        print(f"[spoof.py] Subprocess stdout: {result.stdout}")
        if result.stderr:
            print(f"[spoof.py] Subprocess stderr: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"SimpleITK helper script timed out for {patient_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to run SimpleITK helper script: {e}")

    # Check if the subprocess crashed or returned an error
    if result.returncode != 0:
        error_details = f"Return code: {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        
        # Print the error log if it exists
        if os.path.exists(log_path):
            print(f"[SITK HELPER ERROR LOG for {patient_id}]:")
            with open(log_path, 'r') as f:
                log_content = f.read()
                print(log_content)
                error_details += f"\nLOG FILE: {log_content}"
            os.remove(log_path)
        
        # Clean up files
        if os.path.exists(input_path): 
            os.remove(input_path)
        if os.path.exists(output_path): 
            os.remove(output_path)
        
        raise RuntimeError(f"SimpleITK helper script failed for {patient_id}:\n{error_details}")

    # Check if output file was created
    if not os.path.exists(output_path):
        if os.path.exists(input_path): 
            os.remove(input_path)
        raise RuntimeError(f"SimpleITK helper script did not create output file for {patient_id}")

    # Load the processed tensor from the helper script
    try:
        processed_tensor = torch.load(output_path)
        print(f"[spoof.py] Successfully loaded processed tensor, shape: {processed_tensor.shape}")
    except Exception as e:
        # Clean up files
        if os.path.exists(input_path): 
            os.remove(input_path)
        if os.path.exists(output_path): 
            os.remove(output_path)
        raise RuntimeError(f"Failed to load processed tensor for {patient_id}: {e}")

    # Clean up intermediate files
    try:
        os.remove(input_path)
        os.remove(output_path)
        if os.path.exists(log_path):
            os.remove(log_path)
    except Exception as e:
        print(f"[spoof.py] Warning: Could not clean up intermediate files: {e}")

    return processed_tensor

def get_gradient_magnitude(image_tensor):
    """Computes the gradient magnitude using scipy to avoid SimpleITK."""
    from scipy.ndimage import gaussian_gradient_magnitude
    grad_data = gaussian_gradient_magnitude(image_tensor.numpy(), sigma=1.0)
    return torch.from_numpy(grad_data)

def enhance_contrast_cpu(image_tensor):
    """Applies CLAHE for contrast enhancement slice by slice."""
    img_np = image_tensor.detach().cpu().numpy()
    D, H, W = img_np.shape
    enhanced = np.zeros_like(img_np)
    for d in range(D):
        slice_ = img_np[d, :, :].astype(np.float32)
        if slice_.max() > slice_.min():
            normalized = cv2.normalize(slice_, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced[d] = clahe.apply(normalized)
    return torch.tensor(enhanced).float()

def crop_to_brain_bounding_box(image, label, margin=5):
    """Crops the image and label to the brain's bounding box."""
    nonzero_mask = image[:-1].sum(axis=0) > 0 
    coords = torch.nonzero(nonzero_mask)
    if coords.numel() == 0: return None, None
    min_coords = coords.min(dim=0)[0]
    max_coords = coords.max(dim=0)[0]
    d_min, h_min, w_min = (min_coords - margin).clamp(min=0).tolist()
    d_max, h_max, w_max = (max_coords + margin + 1).clamp(max=torch.tensor(nonzero_mask.shape)).tolist()
    cropped_image = image[:, d_min:d_max, h_min:h_max, w_min:w_max]
    cropped_label = label[d_min:d_max, h_min:h_max, w_min:w_max]
    return cropped_image, cropped_label

# =====================================================================================
# Main Patient Processing Function
# =====================================================================================
def process_patient(patient_path, nyul_normalizer, target_spacing, temp_dir):
    """Main pipeline for processing a single patient."""
    patient_id = os.path.basename(patient_path)
    output_file = os.path.join(save_dir, f"{patient_id}.pt")
    temp_output_file = output_file + ".tmp"

    if os.path.exists(output_file): return "skipped", patient_id
    if os.path.exists(temp_output_file): os.remove(temp_output_file)

    try:
        print(f"\n[DEBUG] Processing patient: {patient_id}")
        # --- 1. Load and Resample/Reorient using nibabel and scipy ---
        try:
            resampled_modalities = []
            for m in modalities:
                filepath = os.path.join(patient_path, f"{patient_id}-{m}.nii.gz")
                print(f"[DEBUG] Loading modality {m} from {filepath}")
                img_nib = nib.load(filepath)
                resampled_data = resample_and_reorient(img_nib, target_spacing, is_label=False)
                print(f"[DEBUG] {m} shape after resample: {resampled_data.shape}, dtype: {resampled_data.dtype}, min: {np.min(resampled_data)}, max: {np.max(resampled_data)}")
                resampled_modalities.append(torch.from_numpy(resampled_data))
        except Exception as e:
            print(f"[EXCEPTION] Error in loading/resampling modalities for {patient_id}: {e}")
            print(traceback.format_exc())
            raise

        try:
            label_filepath = os.path.join(patient_path, f"{patient_id}-seg.nii.gz")
            print(f"[DEBUG] Loading label from {label_filepath}")
            label_nib = nib.load(label_filepath)
            resampled_label_data = resample_and_reorient(label_nib, target_spacing, is_label=True)
            print(f"[DEBUG] Label shape after resample: {resampled_label_data.shape}, dtype: {resampled_label_data.dtype}, min: {np.min(resampled_label_data)}, max: {np.max(resampled_label_data)}")
            label_tensor = torch.from_numpy(resampled_label_data).long()
        except Exception as e:
            print(f"[EXCEPTION] Error in loading/resampling label for {patient_id}: {e}")
            print(traceback.format_exc())
            raise

        try:
            modalities_tensor = torch.stack(resampled_modalities).float()
            print(f"[DEBUG] modalities_tensor shape: {modalities_tensor.shape}, dtype: {modalities_tensor.dtype}, min: {modalities_tensor.min()}, max: {modalities_tensor.max()}")
        except Exception as e:
            print(f"[EXCEPTION] Error stacking modalities for {patient_id}: {e}")
            print(traceback.format_exc())
            raise

        # --- 2. N4 Bias Correction & Anisotropic Diffusion (via isolated subprocess) ---
        try:
            print(f"[DEBUG] Running SimpleITK features (N4 + Anisotropic Diffusion)...")
            image_aniso = run_sitk_features(modalities_tensor, temp_dir, patient_id)
            print(f"[DEBUG] image_aniso shape: {image_aniso.shape}, dtype: {image_aniso.dtype}, min: {image_aniso.min()}, max: {image_aniso.max()}")
        except Exception as e:
            print(f"[EXCEPTION] Error in SimpleITK features for {patient_id}: {e}")
            print(traceback.format_exc())
            raise

        # --- 3. Nyul Histogram Standardization ---
        try:
            print(f"[DEBUG] Running Nyul Histogram Standardization...")
            subject_dict = {m: tio.ScalarImage(tensor=img.unsqueeze(0)) for m, img in zip(modalities, image_aniso)}
            subject = tio.Subject(subject_dict)
            normalized_subject = nyul_normalizer(subject)
            image_nyul = torch.stack([normalized_subject[m].data.squeeze(0) for m in modalities])
            print(f"[DEBUG] image_nyul shape: {image_nyul.shape}, dtype: {image_nyul.dtype}, min: {image_nyul.min()}, max: {image_nyul.max()}")
        except Exception as e:
            print(f"[EXCEPTION] Error in Nyul Histogram Standardization for {patient_id}: {e}")
            print(traceback.format_exc())
            raise

        # --- 4. Enhancement and Gradient Map ---
        try:
            print(f"[DEBUG] Running enhancement and gradient map...")
            image_enhanced = torch.stack([enhance_contrast_cpu(img) for img in image_nyul])
            print(f"[DEBUG] image_enhanced shape: {image_enhanced.shape}, dtype: {image_enhanced.dtype}, min: {image_enhanced.min()}, max: {image_enhanced.max()}")
            grad_map = get_gradient_magnitude(image_enhanced[0]).unsqueeze(0) 
            print(f"[DEBUG] grad_map shape: {grad_map.shape}, dtype: {grad_map.dtype}, min: {grad_map.min()}, max: {grad_map.max()}")
            final_image_to_crop = torch.cat([image_enhanced, grad_map], dim=0)
            print(f"[DEBUG] final_image_to_crop shape: {final_image_to_crop.shape}")
        except Exception as e:
            print(f"[EXCEPTION] Error in enhancement/gradient for {patient_id}: {e}")
            print(traceback.format_exc())
            raise

        # --- 5. Crop to Brain Bounding Box ---
        try:
            print(f"[DEBUG] Cropping to brain bounding box...")
            cropped_image, cropped_label = crop_to_brain_bounding_box(final_image_to_crop, label_tensor)
            if cropped_image is None:
                print(f"[ERROR] Cropping returned None for patient {patient_id} (empty image)")
                return "error_empty_image", patient_id
            print(f"[DEBUG] cropped_image shape: {cropped_image.shape}, cropped_label shape: {cropped_label.shape}")
        except Exception as e:
            print(f"[EXCEPTION] Error in cropping for {patient_id}: {e}")
            print(traceback.format_exc())
            raise

        # --- 6. ATOMIC SAVE ---
        try:
            print(f"[DEBUG] Saving processed tensors...")
            torch.save({"image": cropped_image.half(), "label": cropped_label}, temp_output_file)
            os.rename(temp_output_file, output_file)
            print(f"[DEBUG] Successfully processed and saved patient {patient_id}")
        except Exception as e:
            print(f"[EXCEPTION] Error in saving for {patient_id}: {e}")
            print(traceback.format_exc())
            if os.path.exists(temp_output_file): os.remove(temp_output_file)
            raise

        return "processed", patient_id
    except Exception as e:
        print(f"[EXCEPTION] Error while processing patient {patient_id}: {e}")
        tb_str = traceback.format_exc()
        print(f"[EXCEPTION TRACEBACK]\n{tb_str}")
        if os.path.exists(temp_output_file): os.remove(temp_output_file)
        return "error", patient_id, f"Failed during processing: {e}\n{tb_str}"

# =====================================================================================
# Execution Block
# =====================================================================================
def safe_process(patient_path, nyul_normalizer, target_spacing, temp_dir):
    """Wrapper to catch any exception from the processing function."""
    try:
        return process_patient(patient_path, nyul_normalizer, target_spacing, temp_dir)
    except Exception as e:
        tb_str = traceback.format_exc()
        return "crash", os.path.basename(patient_path), f"CRITICAL FAILURE in process_patient: {e}\n{tb_str}"

if __name__ == "__main__":
    print("Starting preprocessing pipeline...")
    
    all_patients = sorted(glob.glob(os.path.join(data_root, "BraTS-PED-*")))
    print(f"Found {len(all_patients)} patients to process.")

    TARGET_SPACING = get_median_spacing(all_patients)
    print(f"Using dynamically calculated median spacing: {TARGET_SPACING}")

    force_retrain = True
    if force_retrain or not os.path.exists(NYUL_LANDMARKS_PATH):
        if force_retrain and os.path.exists(NYUL_LANDMARKS_PATH):
            print("--- FORCING RETRAINING OF NYUL LANDMARKS (deleting old file) ---")
            os.remove(NYUL_LANDMARKS_PATH)
        else:
            print(f"Nyul landmarks not found. Training on first {NYUL_TRAINING_PATIENT_COUNT} patients...")
        
        training_paths = all_patients[:NYUL_TRAINING_PATIENT_COUNT]
        subjects_for_training = []
        for patient_path in tqdm(training_paths, desc="Loading Nyul Training Data"):
            patient_id = os.path.basename(patient_path)
            t1c_path = os.path.join(patient_path, f"{patient_id}-t1c.nii.gz")
            if os.path.exists(t1c_path):
                subjects_for_training.append(tio.Subject(image=tio.ScalarImage(t1c_path)))
        
        nyul_transform = NyulStandardization(masking_method='otsu')
        landmarks = nyul_transform.train(subjects_for_training)
        np.save(NYUL_LANDMARKS_PATH, landmarks)
        print(f"Nyul landmarks trained and saved to {NYUL_LANDMARKS_PATH}")

    print("Loading pre-trained Nyul landmarks...")
    landmarks = np.load(NYUL_LANDMARKS_PATH)
    
    print("Initializing NyulStandardization...")
    nyul_normalizer = NyulStandardization(landmarks=landmarks, masking_method='otsu')
    print("NyulStandardization initialized successfully.")

    # Clear previous log files
    for log_type in ["processed", "skipped", "errored", "crashed"]:
        log_file = os.path.join(save_dir, f"{log_type}_patients.txt")
        if os.path.exists(log_file): 
            os.remove(log_file)

    print(f"\n--- Starting main processing loop (serial mode) ---")
    
    # --- Process only the first 4 patients for a quick debug run ---
    patients_to_process = all_patients
    print(f"--- RUNNING IN DEBUG MODE: PROCESSING FIRST {len(patients_to_process)} PATIENTS ---")

    results = []
    # Use save_dir for all intermediate files (Google Drive storage)
    print(f"Using Google Drive storage for intermediate files: {save_dir}")
    
    for p in tqdm(patients_to_process, desc="Processing Patients"):
        result = safe_process(p, nyul_normalizer, TARGET_SPACING, save_dir)
        results.append(result)
        # Print immediate result for debugging
        if len(result) >= 2:
            print(f"Result for {result[1]}: {result[0]}")
        else:
            print(f"Unexpected result format: {result}")

    # --- Log the results ---
    processed = {r[1] for r in results if r[0] == "processed"}
    skipped = {r[1] for r in results if "skipped" in r[0]}
    errored = {r[1]: r[2] for r in results if r[0] == "error"}
    crashed = {r[1]: r[2] for r in results if r[0] == "crash"}

    # Write result logs
    with open(os.path.join(save_dir, "processed_patients.txt"), "w") as f:
        for p in sorted(processed): 
            f.write(f"{p}\n")
    
    with open(os.path.join(save_dir, "skipped_patients.txt"), "w") as f:
        for p in sorted(skipped): 
            f.write(f"{p}\n")
    
    with open(os.path.join(save_dir, "errored_patients.txt"), "w") as f:
        for p, err in errored.items(): 
            f.write(f"{p}: {err}\n")
    
    with open(os.path.join(save_dir, "crashed_patients.txt"), "w") as f:
        for p, err in crashed.items(): 
            f.write(f"{p}: {err}\n")

    print(f"\n--- Preprocessing Complete ---")
    print(f"Successfully processed: {len(processed)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Errored: {len(errored)}")
    print(f"Crashed: {len(crashed)}")
    
    if processed:
        print(f"Processed patients: {sorted(processed)}")
    if errored:
        print(f"Errored patients: {sorted(errored.keys())}")
    if crashed:
        print(f"Crashed patients: {sorted(crashed.keys())}")
    
    print(f"Log files written to: {save_dir}")
    print("Processing complete!")
