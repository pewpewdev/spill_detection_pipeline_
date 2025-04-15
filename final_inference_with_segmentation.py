import os
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
import csv


yolo_model_path = 'yolov8m_finetune_wt/yolov8m.pt'
unet_model_path = 'spill_unet_model.h5'
input_image_dir = 'images/val'  
output_image_dir = 'final_test_output_images/'  
csv_file_path = 'test_output_coordinatesX.csv'  


os.makedirs(output_image_dir, exist_ok=True)

# LOAD MODELS 
yolo_model = YOLO(yolo_model_path)
unet_model = load_model(unet_model_path)


def get_center_of_mask(mask):
    coords = cv2.findNonZero((mask > 0.5).astype(np.uint8))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cx = x + w // 2
        cy = y + h // 2
        return (cx, cy)
    return None

#CSV SETUP
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_name', 'spill_location_x', 'spill_location_y'])  # Column headers

#PROCESS ALL IMAGES IN TESTING DIRECTORY 
for image_name in os.listdir(input_image_dir):
    # Check if the file is an image (can be modified to suit file types)
    if image_name.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_image_dir, image_name)
        
        # LOAD IMAGE 
        image = cv2.imread(image_path)
        original = image.copy()
        h_orig, w_orig = image.shape[:2]

        #  YOLO INFERENCE 
        results = yolo_model(image_path, conf=0.2)[0]

        # APPLY NON-MAXIMUM SUPPRESSION (NMS)
        # Filter out detections based on confidence threshold
        results = results.cpu()  # Move to CPU for further processing
        boxes = results.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
        scores = results.boxes.conf  # Confidence scores
        labels = results.boxes.cls  # Class labels

        # NMS operation
        nms_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.2, nms_threshold=0.4)

        # If no boxes are selected, continue with next image
        if len(nms_indices) == 0:
            print(f"⚠️ No valid boxes after NMS in {image_name}.")
            continue

        #  PROCESS ALL DETECTED BOXES AFTER NMS
        for i in nms_indices.flatten():
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box)
            cropped = image[y1:y2, x1:x2]

            #  U-NET PREPROCESS 
            cropped_resized = cv2.resize(cropped, (256, 256))
            input_tensor = cropped_resized / 255.0  # Normalize
            input_tensor = np.expand_dims(input_tensor, axis=0)

            #  PREDICT SEGMENTATION 
            pred_mask = unet_model.predict(input_tensor)[0, :, :, 0]

            #  RESIZE MASK BACK TO CROP SIZE
            pred_mask_resized = cv2.resize(pred_mask, (x2 - x1, y2 - y1))

            #  CREATE BINARY MASK USING LOWER THRESHOLD 
            binary_mask = (pred_mask_resized > 0.1).astype(np.uint8)  # Lowered threshold

            #  FIND CENTER OF MASK 
            center = get_center_of_mask(binary_mask)
            if center:
                cx, cy = center
                cx += x1
                cy += y1

                #  DRAW YOLO BOUNDING BOX AND SPILL LOCATION 
                cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Bounding box with thickness of 4
                cv2.circle(original, (cx, cy), 10, (0, 0, 255), -1)  # Larger red circle for center
                cv2.putText(original, f"spill_location: ({cx}, {cy})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Larger text size and thickness

                #  SAVE SPILL LOCATION COORDINATES TO CSV 
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([image_name, cx, cy])

        #  SAVE FINAL IMAGE 
        output_image_path = os.path.join(output_image_dir, f"output_{image_name}")
        cv2.imwrite(output_image_path, original)  

        print(f"✅ Processed and saved {image_name}.")

print(f"✅ Final images saved in {output_image_dir}")
print(f"✅ Spill location coordinates saved in {csv_file_path}")
