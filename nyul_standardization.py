import torch
import numpy as np
from skimage.filters import threshold_otsu
from torchio.transforms.intensity_transform import IntensityTransform
from torchio.data.subject import Subject

# This is a self-contained implementation of NyulStandardization, adapted from
# older TorchIO versions to be independent of SimpleITK and other internal functions.

def _get_mask_from_tensor(tensor, masking_method):
    print(f"[nyul_standardization.py] _get_mask_from_tensor called with masking_method={masking_method}")
    """
    Computes a binary mask from a tensor using the specified method.
    This version uses skimage instead of SimpleITK.
    """
    if masking_method is None:
        return torch.ones_like(tensor, dtype=torch.bool)

    if not isinstance(masking_method, str):
        # If a custom function is passed
        return masking_method(tensor)

    if masking_method.lower() == 'otsu':
        # Squeeze to handle (1, D, H, W) tensors
        np_tensor = tensor.squeeze().numpy()

        # --- FIX: Add a safeguard for Otsu Thresholding ---
        # Check if the image has variance to prevent crashes on blank images.
        if np_tensor.std() == 0:
            print("[nyul_standardization.py] Zero variance detected, returning zero mask")
            return torch.zeros_like(tensor, dtype=torch.bool)

        try:
            # Use skimage's Otsu thresholding instead of SimpleITK
            threshold = threshold_otsu(np_tensor)
            mask_np = np_tensor > threshold
            print(f"[nyul_standardization.py] Otsu threshold: {threshold}, mask sum: {mask_np.sum()}")
            return torch.from_numpy(mask_np).bool().unsqueeze(0)  # Add channel dim back
        except Exception as e:
            print(f"[nyul_standardization.py] Error in Otsu thresholding: {e}")
            # Fallback to simple thresholding
            mean_val = np_tensor.mean()
            mask_np = np_tensor > mean_val
            return torch.from_numpy(mask_np).bool().unsqueeze(0)
    else:
        raise ValueError(f"Unknown masking method: {masking_method}")


def _apply_landmarks_to_tensor(tensor, image_landmarks, standard_landmarks):
    print("[nyul_standardization.py] _apply_landmarks_to_tensor called")
    """
    Applies histogram landmark-based normalization to a tensor.
    This is a self-contained replacement for torchio's internal _apply_landmarks.
    """
    # Use float32 for interpolation precision
    tensor_np = tensor.numpy().astype(np.float32)
    
    # The mapping is from the image's own landmarks to the standard landmarks
    interp_values = np.interp(tensor_np.ravel(), image_landmarks, standard_landmarks)
    
    return torch.from_numpy(interp_values.reshape(tensor_np.shape))


class NyulStandardization(IntensityTransform):
    """
    Nyúl and Udupa histogram standardization.
    This is a standalone version adapted from TorchIO 0.18.72.
    It does not rely on SimpleITK or other internal functions.
    """
    def __init__(self, landmarks=None, masking_method=None, **kwargs):
        print(f"[nyul_standardization.py] NyulStandardization.__init__ called. landmarks is not None: {landmarks is not None}, masking_method: {masking_method}")
        super().__init__(**kwargs)
        self.landmarks = landmarks
        self.masking_method = masking_method
        self.trained = landmarks is not None

    def train(self, subjects_list):
        print("[nyul_standardization.py] NyulStandardization.train called")
        """Train the model to find the standard landmarks."""
        if not isinstance(subjects_list, (list, tuple)):
            subjects_list = [subjects_list]
        
        all_landmarks = []
        for i, subject in enumerate(subjects_list):
            print(f"[nyul_standardization.py] Training on subject {i+1}/{len(subjects_list)}")
            try:
                image = self.get_image_from_subject(subject)
                mask = _get_mask_from_tensor(image.data, self.masking_method)
                
                values = image.data[mask].numpy().astype(np.float32)
                print(f"[nyul_standardization.py] Subject {i+1}: {values.shape[0]} masked values")
                
                if values.size == 0:
                    print(f"[nyul_standardization.py] Warning: No masked values for subject {i+1}, skipping")
                    continue
                    
                percentiles = np.linspace(0, 100, num=11) # Standard 11 landmarks
                landmarks = np.percentile(values, percentiles)
                all_landmarks.append(landmarks)
                print(f"[nyul_standardization.py] Subject {i+1} landmarks: {landmarks}")
                
            except Exception as e:
                print(f"[nyul_standardization.py] Error processing subject {i+1}: {e}")
                continue
                
        if not all_landmarks:
            raise RuntimeError("No valid subjects found for training Nyul landmarks")
            
        self.landmarks = np.mean(all_landmarks, axis=0)
        self.trained = True
        print(f"[nyul_standardization.py] Final trained landmarks: {self.landmarks}")
        return self.landmarks

    def apply_transform(self, subject: Subject) -> Subject:
        print("[nyul_standardization.py] NyulStandardization.apply_transform called")
        if not self.trained:
            raise RuntimeError("The NyulStandardization transform has not been trained.")
            
        for image in self.get_images(subject):
            print("[nyul_standardization.py]  Processing image in subject...")
            try:
                mask = _get_mask_from_tensor(image.data, self.masking_method)
                print(f"[nyul_standardization.py]   Mask shape: {mask.shape}, mask sum: {mask.sum().item()}")
                
                # Calculate landmarks for the current image
                values = image.data[mask].numpy().astype(np.float32)
                print(f"[nyul_standardization.py]   Values shape: {values.shape}, min: {values.min() if values.size > 0 else 'NA'}, max: {values.max() if values.size > 0 else 'NA'}")
                
                if values.size == 0:
                    print("[nyul_standardization.py]   Warning: No masked values, skipping normalization")
                    continue
                    
                percentiles = np.linspace(0, 100, num=11)
                image_landmarks = np.percentile(values, percentiles)
                print(f"[nyul_standardization.py]   Image landmarks: {image_landmarks}")

                # Apply the mapping from image landmarks to standard landmarks
                normalized_data = _apply_landmarks_to_tensor(image.data, image_landmarks, self.landmarks)

                # Put the non-masked values back
                final_data = image.data.clone()
                final_data[mask] = normalized_data[mask]
                image.set_data(final_data)
                print("[nyul_standardization.py]   Image normalization complete.")
                
            except Exception as e:
                print(f"[nyul_standardization.py]   Error in normalization: {e}")
                # Continue without modification if normalization fails
                continue
            
        return subject

    def get_image_from_subject(self, subject):
        if isinstance(subject, Subject):
            return subject.get_first_image()
        else:
            # Handle cases where just an image might be passed
            return subject