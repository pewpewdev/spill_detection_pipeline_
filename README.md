# spill_detection_pipeline_

This repository provides a pipeline for detecting spills in images using YOLOv8, YOLOv5 and then cross verify to get the location of the spill (center) usin a UNet model. The workflow includes annotation conversion, training separate YOLO models for spill and then training a UNet for segmenting the spill detected by yolo, and running a two-stage inference : first inference with YOLO model then with UNet model.

Project structure

📦 SpillDetection/ 
├── images |
├── tarin/ # input images for YOLO model to train 
│ ├── val/ 
├── labels/ #yolo labels (txt) 
| ├── tarin/
│ ├── val/
├── images_seg/
│ ├── images/ # Input images for UNet segmentation 
│ ├── label_json/ # COCO-style JSON labels 
│ └── masks/ # Binary masks for UNEt 
├── image preprocessing.ipynb #image augmentation for YOLO 
├── inference yolo.ipynb #for inferencing with only YOLO model 
├── final_inference_with_segmentation.py #for inferencing with both YOLO & Unet 
├── yolov8m_training.ipynb #Yolov8 model training 
├── yolov8m.pt (primary model) #Primary model weight 
├── yolov5n_training.ipynb #Yolov5 model training 
├── yolov5n.pt (lightweight model) #lightweight model weight 
├── train_segmentation_model.ipynb #UNetmodel training 
├── spill_unet_model.h5 #U-net model weight (
├── detection_results_yolov8.csv #result using only yolo model
├── detection_results_yolov5.csv
├── inference_output_v8finetuned/ #result images wth only YOLO 
├── test_output_coordinates.csv #result using both YOLO and UNet 
└── final_test_output_images/ #result images wth both YOLO and UNet

Environment Setup
Install the required dependencies:

pip install -r requirements.txt


## Model Weights: "https://drive.google.com/drive/u/0/folders/1IzsKfUmNJ3mo68A4eN9kAR-0TyfhWm0t"

## Outputs

| File / Folder                           | Description                                      |
|-----------------------------------------|--------------------------------------------------|
| `detection_results_yolov8.csv`          | Coordinates from YOLO-only detections            |
| `test_output_coordinates.csv`          | Center coordinates from YOLO + UNet              |
| `inference_output_v8finetuned/`         | Images with YOLO predictions                     |
| `final_test_output_images/`             | Final output images after full pipeline          |


## 📘 Documentation

Detailed explanation of the approach, architecture, and training steps is available in the "[ spill_detection_documentation]" 


##  Run Full Inference

To run the full pipeline (YOLO ➡️ UNet ➡️ Coordinates):

```bash
python "final inference with segmentation.py"


## Pipeline Overview

### Data Preparation

- Place training/validation images and YOLO-format labels in their respective folders.
- For segmentation, prepare images and corresponding masks in `images_seg/`.

### Model Training

- Use `train segmentation model.ipynb` to train the UNet segmentation model.
- YOLO models (`yolov8m.pt`, `yolov5n.pt`) can be trained using `yolov8m_training.ipynb`    and `yolov5n_training.ipynb`   accordingly

### Image Augmentation

- Run `image_preprocessing.ipynb` for augmenting images before training.

### Inference (YOLO Only)

- Use `inference yolo.ipynb` to run detection and save results using YOLO only.

### Final Inference with YOLO + UNet

- Run `final inference with segmentation.py` for full pipeline inference.
- This script will:
    - Use YOLO to detect spills.
    - Crop and pass detected regions to the UNet.
    - Output refined masks and calculate center coordinates.


