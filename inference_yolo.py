import os
import cv2
import torch
import pandas as pd
from pathlib import Path
from ultralytics import YOLO  # For YOLOv8

# Configuration
model_path = 'yolov8m_finetune_wt/yolov8m.pt'  
val_dir = 'images_/val'       
output_dir = 'inference_output_v8finetuned'     
os.makedirs(output_dir, exist_ok=True)

# Load the YOLOv8 model
model = YOLO(model_path)

# Store all results
results_list = []

# Process each image in validation folder
for img_file in os.listdir(val_dir):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(val_dir, img_file)
        image = cv2.imread(img_path)
        height, width, _ = image.shape

        # Run inference
        results = model(img_path, conf=0.5)[0]  # Increased confidence threshold to 0.5


        # Parse results
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Calculate center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw center point
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

            # Label and location text
            label_text = f'{label} ({conf:.2f})'
            location_text = f'Spill location: ({cx}, {cy})'

            # Draw label text above bounding box
            cv2.putText(image, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw location text below bounding box
            text_y = y2 + 20 if (y2 + 20) < height else y2 - 10
            cv2.putText(image, location_text, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Store results
            results_list.append({
                'image': img_file,
                'label': label,
                'confidence': round(conf, 4),
                'bbox_x1': x1,
                'bbox_y1': y1,
                'bbox_x2': x2,
                'bbox_y2': y2,
                'center_x': cx,
                'center_y': cy
            })

        # Save annotated image
        output_img_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_img_path, image)

# Save results to CSV
df = pd.DataFrame(results_list)
df.to_csv(os.path.join(output_dir, 'detection_results_yolov8.csv'), index=False)

print(f"Inference complete. Annotated images and CSV saved to '{output_dir}'")
