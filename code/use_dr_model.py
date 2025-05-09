import os
import numpy as np
import tensorflow as tf
# Fix imports based on your TensorFlow version
try:
    # For newer TensorFlow versions
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
except ImportError:
    # For older TensorFlow versions
    from tensorflow.python.keras.models import load_model
    from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration parameters
IMG_SIZE = 224  # Ensure this matches the size used during training

# Class names for diabetic retinopathy severity
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def load_pretrained_model(model_path):
    """Load the pretrained diabetic retinopathy model."""
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            print(f"Model file not found at path: {model_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Files in directory where model should be: {os.listdir(os.path.dirname(model_path) if os.path.dirname(model_path) else '.')}")
            return None
            
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def preprocess_image(image_path):
    """Preprocess image for prediction."""
    try:
        # Load and resize image
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        
        # Convert to array and normalize
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        # Expand dimensions to match model input shape
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(model, image_path):
    """Make prediction using the model."""
    try:
        # Preprocess the image
        processed_img = preprocess_image(image_path)
        
        if processed_img is None:
            return "Error", [0, 0, 0, 0, 0]
        
        # Make prediction
        predictions = model.predict(processed_img)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        
        # Calculate severity percentage (scale from 0-4 to 0-100%)
        # No DR (0) = 0%, Proliferative DR (4) = 100%
        severity_percentage = (predicted_class_index / 4) * 100
        
        # Get prediction probabilities as percentages
        prediction_percentages = [float(p) * 100 for p in predictions[0]]
        
        return predicted_class, severity_percentage
    except Exception as e:
        print(f"Error making prediction: {e}")
        return "Error", 0