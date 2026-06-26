import os
import datetime
import time
import hashlib
import numpy as np
import tensorflow as tf
from typing import Dict, Any

from src.utils.logger import get_logger

class DiseaseDetectionAgent:
    def __init__(self, model_path: str = "models/crop_disease_mobilenetv2.keras", class_names_path: str = "models/class_names.npy"):
        """
        Initializes the Disease Detection Agent by loading the pre-trained model and class names.
        """
        self.logger = get_logger("disease_agent", "disease_agent.log")
        self.model_version = "crop_disease_mobilenetv2_v1"
        self.model_path = model_path
        
        # Load the model
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Hash model
        with open(model_path, "rb") as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
            
        self.logger.info(f"Loading TensorFlow {tf.__version__}")
        self.logger.info(f"CUDA available: {tf.test.is_built_with_cuda()}, GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        self.logger.info(f"Model Path: {model_path}, SHA256: {model_hash}")
        
        start_load = time.time()
        self.model = tf.keras.models.load_model(model_path)
        
        self.logger.info(f"Model loaded in {int((time.time() - start_load) * 1000)}ms. Output classes: {self.model.output_shape[-1]}")
        
        # Load class names
        if not os.path.exists(class_names_path):
            self.logger.error(f"Class names file not found at {class_names_path}")
            raise FileNotFoundError(f"Class names file not found at {class_names_path}")
        self.class_names = np.load(class_names_path, allow_pickle=True)
        
        # MobileNetV2 standard input size
        self.target_size = (224, 224)
        self.logger.info(f"Input size: {self.target_size}")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocesses an image for MobileNetV2 prediction.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        # Load image with target size
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Expand dimensions to match expected (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)
        
        # MobileNetV2 preprocessing: scale pixels between -1 and 1
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Predicts the disease for the given image path and returns formatted results.
        """
        try:
            self.logger.info(f"Image preprocessing started for {image_path}")
            processed_img = self.preprocess_image(image_path)
            
            self.logger.info("TensorFlow inference started")
            inference_start = time.time()
            # Get raw probabilities (the model outputs softmax probabilities)
            predictions = self.model.predict(processed_img, verbose=0)[0]
            inference_time_ms = int((time.time() - inference_start) * 1000)
            self.logger.info(f"TensorFlow inference ended in {inference_time_ms}ms")
        except Exception as e:
            self.logger.error(f"Exception during prediction: {e}", exc_info=True)
            raise
        
        # Get top 3 indices in descending order
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        
        top_predictions = []
        for idx in top_3_indices:
            confidence_pct = round(float(predictions[idx]) * 100, 2)
            top_predictions.append({
                "class": str(self.class_names[idx]),
                "confidence": confidence_pct
            })
            
        self.logger.info(f"Top 3 predictions: {top_predictions}")
        top_disease = top_predictions[0]["class"]
        top_confidence = top_predictions[0]["confidence"]
        
        # Determine confidence level
        if top_confidence >= 90.0:
            confidence_level = "High"
        elif top_confidence >= 70.0:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
            
        result = {
            "disease": top_disease,
            "confidence": top_confidence,
            "confidence_level": confidence_level,
            "top_predictions": top_predictions,
            "timestamp": datetime.datetime.now().replace(microsecond=0).isoformat(),
            "model_version": self.model_version
        }
        
        # Add a warning if confidence is low
        if confidence_level == "Low":
            result["warning"] = "Low confidence prediction. Please verify visually or retake the image."
            
        self.logger.info(f"Returned JSON: {result}")
        return result
