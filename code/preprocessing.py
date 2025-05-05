import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import exposure, filters
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.filters import frangi, hessian, sato

def preprocess_fundus_image(image_path, output_size=(224, 224)):
    # Read image
    img = cv2.imread(str(image_path))
    
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Create circular mask to isolate ROI
    height, width = gray.shape
    center = (width // 2, height // 2)
    
    # Find the radius of the circular region
    mask = np.zeros_like(gray)
    x, y = np.ogrid[:height, :width]
    
    # Find non-zero pixels to determine the circle
    non_zero = np.where(gray > 10)
    if len(non_zero[0]) > 0:
        radius = min(width, height) // 2 - 10  # Give some margin
        circle_mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        mask[circle_mask] = 255
    
    # Apply mask to the image
    masked_img = cv2.bitwise_and(gray, gray, mask=mask)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked_img)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Apply vessel enhancement filter (using Hessian matrix)
    
    vessel_filter = frangi(blurred)
    
    # Normalize to [0, 255]
    vessel_filter = ((vessel_filter - vessel_filter.min()) * 255 / 
                    (vessel_filter.max() - vessel_filter.min())).astype(np.uint8)
    
    # Combine the enhanced image with vessel map
    combined = cv2.addWeighted(enhanced, 0.7, vessel_filter, 0.3, 0)
    
    # Resize to the target size
    resized = cv2.resize(combined, output_size)
    
    # Normalize pixel values to [0, 1]
    normalized = resized / 255.0
    
    return normalized

def preprocess_dataset(input_dir, output_dir, output_size=(224, 224)):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    image_paths = list(input_path.glob('*.jpg')) + list(input_path.glob('*.jpeg')) + list(input_path.glob('*.png'))
    
    for img_path in image_paths:
        preprocessed = preprocess_fundus_image(img_path, output_size)
        
        # Save as numpy array or image
        output_file = output_path / f"{img_path.stem}_preprocessed.npy"
        np.save(output_file, preprocessed)
        
        # Alternatively, save as image
        output_img = (preprocessed * 255).astype(np.uint8)
        cv2.imwrite(str(output_path / f"{img_path.stem}_preprocessed.png"), output_img)
        
    print(f"Preprocessed {len(image_paths)} images")

# Example usage
# preprocess_dataset('path/to/input/images', 'path/to/output/preprocessed')

def create_augmentation_generator():
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,  # Usually not flipped vertically for medical images
        fill_mode='nearest'
    )
    return datagen

if __name__ == "__main__":
    base_input_dir = "./content/Diabetic_Balanced_Data"
    base_output_dir = "./content/Preprocessed_Diabetic_Data"
    
    # Process each class (0-4) in each dataset split (test, train, val)
    for split in ["train", "val"]:
        for class_num in range(5):  # 0-4 classes
            if(split == "train" and class_num < 4):\
                pass
            else:
                input_path = f"{base_input_dir}/{split}/{class_num}"
                output_path = f"{base_output_dir}/{split}/{class_num}"
                print(f"Processing {input_path} -> {output_path}")
                preprocess_dataset(input_path, output_path)