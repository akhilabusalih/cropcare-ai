import os
import sys
import argparse
import csv
import logging
from glob import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agents.disease_detection_agent.disease_detection_agent import DiseaseDetectionAgent
from src.evaluation.metrics import (
    calculate_basic_metrics,
    generate_classification_report,
    generate_confusion_matrix,
    get_class_performance,
    get_top_confused_pairs
)
from src.evaluation.report_generator import (
    save_json_metrics,
    save_classification_report,
    save_confusion_matrix_plot,
    generate_model_health_report
)

def evaluate_dataset(dataset_dir, prefix, output_dir="reports"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Initializing DiseaseDetectionAgent...")
    # Initialize the agent
    agent = DiseaseDetectionAgent()
    target_names = list(agent.class_names)
    
    y_true = []
    y_pred = []
    misclassified = []
    
    logger.info(f"Evaluating images in {dataset_dir}...")
    
    # Iterate through all class subdirectories
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        if class_name not in target_names:
            logger.warning(f"Directory '{class_name}' is not in known class names. Skipping.")
            continue
            
        true_label_idx = target_names.index(class_name)
        
        # Supported extensions
        extensions = ('*.jpg', '*.jpeg', '*.png')
        images = []
        for ext in extensions:
            images.extend(glob(os.path.join(class_dir, ext)))
            images.extend(glob(os.path.join(class_dir, ext.upper())))
            
        for img_path in images:
            try:
                result = agent.predict(img_path)
                pred_class_name = result["disease"]
                confidence = result["confidence"]
                
                if pred_class_name in target_names:
                    pred_label_idx = target_names.index(pred_class_name)
                else:
                    # Should not happen if agent works correctly
                    pred_label_idx = -1 
                    
                y_true.append(true_label_idx)
                y_pred.append(pred_label_idx)
                
                if true_label_idx != pred_label_idx:
                    misclassified.append({
                        "image_path": img_path,
                        "predicted_class": pred_class_name,
                        "actual_class": class_name,
                        "confidence": confidence
                    })
                    
            except Exception as e:
                logger.error(f"Error predicting {img_path}: {e}")

    if not y_true:
        logger.warning("No valid images found for evaluation.")
        return

    # 1. Metrics Calculation
    logger.info("Calculating metrics...")
    basic_metrics = calculate_basic_metrics(y_true, y_pred)
    class_report_str = generate_classification_report(y_true, y_pred, target_names)
    cm = generate_confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
    strongest, weakest = get_class_performance(y_true, y_pred, target_names)
    top_confused = get_top_confused_pairs(y_true, y_pred, target_names)
    
    # 2. Saving Reports
    logger.info(f"Saving reports to {output_dir}...")
    
    save_json_metrics(basic_metrics, os.path.join(output_dir, f"{prefix}_metrics.json"))
    save_classification_report(class_report_str, os.path.join(output_dir, f"{prefix}_classification_report.txt"))
    save_confusion_matrix_plot(cm, target_names, os.path.join(output_dir, f"{prefix}_confusion_matrix.png"))
    
    # Write misclassified CSV
    csv_path = os.path.join(output_dir, f"{prefix}_misclassified_images.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "predicted_class", "actual_class", "confidence"])
        writer.writeheader()
        writer.writerows(misclassified)
        
    # Generate Model Health Report
    generate_model_health_report(
        os.path.join(output_dir, f"{prefix}_model_health_report.md"),
        overall_accuracy=basic_metrics["accuracy"],
        strongest_classes=strongest,
        weakest_classes=weakest,
        top_confused_pairs=top_confused
    )
    
    logger.info(f"Evaluation complete for {prefix}. Reports saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Disease Detection Agent")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to test dataset (e.g., data/test)")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix for report files (e.g., plantvillage or realworld)")
    parser.add_argument("--output_dir", type=str, default="reports", help="Output directory for reports")
    
    args = parser.parse_args()
    evaluate_dataset(args.dataset_dir, args.prefix, args.output_dir)
