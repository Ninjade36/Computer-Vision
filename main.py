#!/usr/bin/env python3


from PIL import Image, ImageOps
import argparse
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import os
import torch
from objRemove import ObjectRemove
from models2.deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# Set the paths
model_path = r'C:\\Users\\Adi\\Desktop\\fourthyear\\DL-Proj\\object-remove-main\\src\\models2\\best.pt'
deepfill_weights_path = 'C:/Users/Adi/Desktop/fourthyear/DL-Proj/object-remove-main/src/models2/states_pt_places2.pth' 
output_dir = "static\output"

# Load YOLOv5 and DeepFill models
model_custom = torch.hub.load(r'C:\Users\Adi\Desktop\fourthyear\DL-Proj\object-remove-main\yolov5', 'custom', path=model_path, source='local', force_reload=True)
model_custom.eval()
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
rcnn_transforms = weights.transforms()
rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False)
rcnn.eval()
deepfill = Generator(checkpoint=deepfill_weights_path, return_flow=True)

def get_detections2(image_path, model, conf=0.5):
    results = model(image_path)
    results = results.pandas().xyxy[0]
    detections = results[results['confidence'] > conf]
    return detections[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

def get_detections(image_path, model, conf=0.5):
    results = model(image_path)
    results = results.pandas().xyxy[0]  # Results as a DataFrame
    detections = []
    for index, row in results.iterrows():
        if row['confidence'] > conf:
            class_name = row['name']  # Get the class name
            detections.append({'xmin': row['xmin'], 'ymin': row['ymin'], 'xmax': row['xmax'], 'ymax': row['ymax'], 'class': class_name})
    return detections

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, bbox in enumerate(params['detections']):
            xmin, ymin, xmax, ymax = bbox
            if xmin <= x <= xmax and ymin <= y <= ymax:
                print(f"Car {i+1} selected for removal")
                params['selected_bbox'] = bbox
                cv2.destroyAllWindows()
                break

def display_and_select(image_path, detections):

    img = cv2.imread(image_path)
    for i, (xmin, ymin, xmax, ymax) in enumerate(detections):
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(img, f"Car {i+1}", (int(xmin), int(ymin-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    params = {'detections': detections, 'selected_bbox': None}
    cv2.imshow("Select a car to remove by clicking on it", img)
    cv2.setMouseCallback("Select a car to remove by clicking on it", click_event, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return params['selected_bbox']

def resize_image(image_path, target_size=(512, 680), output_format='JPEG'):
    image = Image.open(image_path)
    image = ImageOps.contain(image, target_size, method=Image.Resampling.LANCZOS)
    filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{filename_without_ext}_resized.{output_format.lower()}")
    image.save(output_path, format=output_format)
    return output_path

def run_object_removal(image_path, detection):
    detection_format = [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']]
    print("im the detection format",detection_format)
    model = ObjectRemove(segmentModel=rcnn, rcnn_transforms=rcnn_transforms, inpaintModel=deepfill, image_path=image_path, detections=[detection_format])
    output = model.run()
    output_file_path = os.path.join(output_dir, 'output_image.jpg')
    cv2.imwrite(output_file_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    return output_file_path


def process_image(image_path,detections):
    
    image_path=r'static\\'+image_path
    
    selected_bbox = detections

    if selected_bbox:
        output_path = run_object_removal(image_path, selected_bbox)
        print(f"Processed image saved at {output_path}")
        return os.path.join('output', os.path.basename(output_path))
    else:
        print("No car was selected for removal.")
