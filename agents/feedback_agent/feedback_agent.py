import os
import csv
import shutil
import hashlib
from collections import Counter
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger("feedback_agent", "feedback_agent.log")

class FeedbackAgent:
    def __init__(self, feedback_dir: str = "data/feedback"):
        self.feedback_dir = feedback_dir
        self.images_dir = os.path.join(feedback_dir, "images")
        self.csv_path = os.path.join(feedback_dir, "feedback_log.csv")
        self.fieldnames = [
            "image_path",
            "predicted_class",
            "actual_class",
            "confidence",
            "timestamp",
            "model_version"
        ]
        
        # Ensure directories exist
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def _get_image_hash(self, image_path: str) -> str:
        """Returns the MD5 hash of an image file to check for duplicates."""
        hasher = hashlib.md5()
        with open(image_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def log_feedback(self, 
                     original_image_path: str, 
                     predicted_class: str, 
                     actual_class: str, 
                     confidence: float, 
                     timestamp: str, 
                     model_version: str) -> bool:
        """
        Logs feedback, copies the image securely, and prevents duplicates.
        Returns True if logged successfully, False if it was a duplicate.
        """
        if not os.path.exists(original_image_path):
            raise FileNotFoundError(f"Image not found at {original_image_path}")

        img_hash = self._get_image_hash(original_image_path)
        ext = os.path.splitext(original_image_path)[1]
        if not ext:
            ext = ".jpg"
            
        new_filename = f"{img_hash}{ext}"
        new_image_path = os.path.join(self.images_dir, new_filename)
        
        # Check if hash already exists in our stored images
        if os.path.exists(new_image_path):
            logger.warning(f"Duplicate feedback detected for hash {img_hash}. Ignoring.")
            return False # Duplicate detected
            
        # Copy image
        shutil.copy2(original_image_path, new_image_path)
        logger.info(f"Feedback image copied to {new_image_path}")
        
        # Determine relative path for CSV or just basename
        csv_image_path = new_filename
        
        # Append to CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({
                "image_path": csv_image_path,
                "predicted_class": predicted_class,
                "actual_class": actual_class,
                "confidence": confidence,
                "timestamp": timestamp,
                "model_version": model_version
            })
            
        logger.info(f"Feedback logged to CSV: {predicted_class} -> {actual_class}")
        return True

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Calculates statistics from the feedback log."""
        if not os.path.exists(self.csv_path):
            return {
                "total_feedback": 0,
                "most_confused_class": None,
                "average_wrong_confidence": 0.0
            }
            
        total_feedback = 0
        total_confidence = 0.0
        confused_classes = Counter()
        
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_feedback += 1
                try:
                    conf = float(row["confidence"])
                except ValueError:
                    conf = 0.0
                total_confidence += conf
                
                # We define "most confused class" as the class the model got wrong most often
                # i.e., the actual class that was mispredicted.
                confused_classes[row["actual_class"]] += 1
                
        if total_feedback == 0:
            return {
                "total_feedback": 0,
                "most_confused_class": None,
                "average_wrong_confidence": 0.0
            }
            
        avg_conf = total_confidence / total_feedback
        most_confused = confused_classes.most_common(1)[0][0] if confused_classes else None
        
        return {
            "total_feedback": total_feedback,
            "most_confused_class": most_confused,
            "average_wrong_confidence": round(avg_conf, 2)
        }

    def export_retraining_dataset(self, output_dir: str = "data/retraining_dataset") -> int:
        """
        Exports all logged feedback images into a folder structure suitable for retraining:
        output_dir/actual_class/image.jpg
        Returns the number of images exported.
        """
        if not os.path.exists(self.csv_path):
            return 0
            
        exported_count = 0
        
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                actual_class = row["actual_class"]
                filename = row["image_path"]
                src_path = os.path.join(self.images_dir, filename)
                
                if os.path.exists(src_path):
                    class_dir = os.path.join(output_dir, actual_class)
                    os.makedirs(class_dir, exist_ok=True)
                    
                    dst_path = os.path.join(class_dir, filename)
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                        exported_count += 1
                        
        logger.info(f"Exported {exported_count} images to {output_dir}")
        return exported_count
