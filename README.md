# Computer-Vision

Sure! Here's the description translated into English:

---

## Applied Learning Project - Object Detection and Removal

### Project Background
Our research focuses on developing advanced methods for digital image editing, specifically through automating the processes of object detection and removal from images. The motivation behind the project is to learn and gain hands-on experience with large datasets, train deep learning models, and integrate them with other models we have no prior knowledge of. Additionally, we aim to improve efficiency and accuracy in the field while enhancing the user experience by minimizing the number of actions needed to achieve their goals. These processes enhance the usability of images for various purposes, from aesthetic improvements to privacy preservation and the development of machine learning systems. The traditional approach, based on manual work and reliant on the operator's skill, is inefficient and resource-intensive. Our project addresses these challenges by exploring the ability to identify and remove specific objects, such as vehicles, from digital images, as an initial step before expanding to other objects.

### Project Description
Our project consists of two main parts:
1. Object Detection in Images
2. Object Removal from Images

#### Object Detection in Images
1. **Data:**
   For the project, we chose to use the COCO-2017 dataset. The choice of COCO-2017 is based on the diversity of images containing objects in different contexts, making it ideal for training models that can understand and identify objects in images. Additionally, we used the FiftyOne library for downloading and preprocessing the data, detailed later.

   The dataset included 6000 training images and 2000 validation images. Due to processing power constraints, we decided to limit the dataset to images with vehicles only (cars, buses, trucks, motorcycles), and the library limited the validation set to 796 images.

   During the initial testing phase, the process of downloading and preprocessing the images was done manually. However, we identified that many images were not suitable for our research needs as they did not include the objects of interest – vehicles, which were supposed to be the focus of the model's learning. Addressing this challenge led us to explore and evaluate additional, more efficient methods for completing the task. During our research, we came across the FiftyOne library, which became a significant tool in our project.

   The FiftyOne library provides an advanced solution for downloading and preprocessing images, simplifying these processes significantly. It allows users to precisely define which types of objects will appear in the dataset, thereby controlling the relevance and quality of the received data. Additionally, the library supports the ability to visualize the images, allowing us to assess the raw material before starting the prolonged model training process.

2. **Preprocessing:**
   The preprocessing stage of the data in this project included two main steps:
   1. **Data Format Conversion:** The first task was to convert the dataset from the COCO format, based on JSON, to the YOLO format, which uses TXT files. Initially, this process was done manually (see attached file: converting-try1), where the images were saved in their original format, each accompanied by a suitable TXT file detailing the specific objects identified in the image. However, after discovering advanced functionality in the FiftyOne package that allows for automatic and rapid conversion to YOLOv5 format, we abandoned the manual method in favor of the automatic one, saving considerable time and effort. These steps are documented in the training notebook Dl-proj-10k.ipynb.
   2. **Data Cleaning:** Following the initial training phase, unsatisfactory performance was identified, partly due to the presence of irrelevant objects in the images that were not included in the selected vehicle categories. These objects caused the model to focus on incorrect elements, thereby reducing detection accuracy. To solve this problem, we decided to go through the conversion process again, this time defining all non-vehicle objects as class=other and then removing them, preventing their inclusion in the training data. This action helped to focus the model's learning and improve its ability to accurately identify relevant vehicles.

3. **Training the Object Detection Model:**
   For the project, we chose YOLOv5, the fifth version of the You Only Look Once series, as the main model for real-time object detection. It is one of the newest and most advanced technologies in the field, offering significant improvements in detection speed and accuracy compared to previous versions. YOLOv5's popularity among computer vision developers can be explained by the following advantages:
   1. **Speed and Efficiency:** Processing images at high speed while maintaining maximum accuracy makes YOLOv5 suitable for real-time applications and environments with limited computing resources.
   2. **High Accuracy:** The model can accurately identify objects even in complex conditions, such as images with overlapping objects and complicated backgrounds.
   3. **Flexibility and Accessibility:** The ease of use and suitability for a wide range of detection tasks, with support for custom training on specific datasets.
   4. **Wide Community Support:** Supported by a large and active community of developers and data scientists, providing numerous resources, tools, and add-ons.

**Model Adaptation for Vehicle Detection:**
   We decided to train the YOLOv5 model on our specific dataset to optimize it for detecting vehicles in images. Through custom training, the model developed a deep understanding of the unique features of vehicles, increasing detection accuracy and reducing the likelihood of errors. The goal was to enhance the model's ability to accurately and quickly identify vehicles regardless of various conditions or vehicle types.

**Model Tuning:**
   We started training the model with an image size of 640, a batch size of 8, and 100 epochs. We achieved an average precision and recall of 65%.

   Therefore, we improved the image size to 720 and then to 1024, with a batch size of 16, and applied early stopping to reduce runtime in case of no improvement, reaching an average of 75%.

   It is important to note that our data was unbalanced between classes: cars made up 60% of the cases, 20% for buses, 20% for trucks, and 20% for motorcycles, resulting in relatively low accuracy due to initial focus on cars. However, overall, when we tested the model on new images, it performed perfectly as required.

**Model Performance:**

1. **Training and Validation Loss:** The box loss, object loss, and class loss consistently decreased during training, indicating model improvement and convergence. The validation loss was higher compared to training but also decreased over time, suggesting good model generalization.
2. **Precision and Recall:** Precision and recall during training were stable with a slight improvement trend, around 0.65 for precision and 0.60 for recall. For validation, we saw an increase in precision and recall, trending towards convergence around 0.675 and 0.625, respectively.
3. **mAP:** The mean average precision (mAP) during training and validation improved consistently, reaching approximately 0.50 in training and 0.45 in validation for mAP with a threshold range of 0.5 to 0.95.
4. **Recall-Confidence Curve:** The curve shows that for high confidence levels, the recall for all classes was high, converging to a value of 0.83 at a confidence level of 0.000.
5. **Precision-Recall Curve:** The model showed high precision for the bus class with a value of 0.831, while the truck class had the lowest precision with a value of 0.554. The average precision for all classes was 0.694 at a threshold resolution of 0.5.
6. **Precision-Confidence Curve:** Precision improved as the confidence level increased, reaching a perfect value of 1.00 at a confidence level of 0.987 for all classes.
7. **F1-Confidence Curve:** This curve shows the harmonic mean of precision and recall for different vehicle types. The highest score for buses is 0.831, indicating consistent and accurate detection. For all classes, the optimal score is 0.69 at a confidence level of 0.258.

Overall, our model demonstrates good performance with particularly strong detection capabilities for specific classes like vehicles and buses, maintaining high accuracy with varying confidence levels despite time and cost constraints. Training on 6000 out of 120,000 images, our model surpassed pre-trained models' performance, and we are satisfied with the results.

We will provide an example comparing the performance of our model against the largest and most trained YOLOv5XL model. In the first image, our model detects vehicles with higher probabilities and successfully captures more complex objects.

**Hardware and Training Costs:**
1. CPU-Tesla V100
2. CPU-Nvidia A100
3. RAM 50-80 GB
4. Computing Units+Drive Storage = 80 Shekel

#### Object Removal from Images
**Object Filling Model:**
After focusing on the custom YOLOv5 model, we continued the process using Mask R-CNN, which allowed us to accurately identify the objects in the image and separate them from the background. Each object identified by the model was given a segmentation mask describing its precise shape. At this stage, we used two important functions to analyze the results: the `iou` function and the `percent_within` function.

The `iou` (Intersection over Union) function calculates the IOU between each pair of boxes, serving as a central tool for assessing object detection accuracy by calculating their overlap. This function allowed us to accurately measure the model's detection quality and compare different generated masks.

The `percent_within` function calculates the percentage of points within a given rectangle, enabling us to assess how much of the object or mask is within the desired area. Using this function was critical for analyzing the mask data and understanding the model's ability to accurately focus on objects.

After obtaining accurate segmentation masks, the process continued with the use of DeepFill technology, designed to fill the gaps created after object removal from the image. Using Generative Advers

arial Networks (GANs) and deep learning methods, DeepFill reconstructed the missing parts of the image convincingly and consistently, creating seamless results without a trace of the removed object. With the help of advanced technologies such as DeepFill and our precise YOLOv5 and Mask R-CNN models, we successfully achieved the project's goal – efficient and accurate removal of objects from images while maintaining high visual quality.

### Results and Achievements
1. Efficiently automated the process of identifying and removing vehicles from images.
2. Developed an accurate object detection model based on YOLOv5.
3. Successfully filled gaps created by object removal using DeepFill technology.

### Conclusion and Future Directions
Our project demonstrates the potential of using advanced deep learning techniques for automating complex image editing tasks. In the future, we aim to expand our research to include other object types and improve the efficiency and accuracy of our models. Additionally, we plan to explore real-time applications of our technology in various fields such as security, automotive, and healthcare.
