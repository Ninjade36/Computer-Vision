### Deep Learning Project: Object Detection and Removal

#### Project Background
My project focuses on developing advanced methods for editing digital images by automating the processes of object detection and removal. The motivation behind this project was to gain hands-on experience with large datasets, train deep learning models, and integrate them with additional models. The aim was to enhance efficiency and accuracy in the field while improving user experience by minimizing the number of actions required to achieve desired results. These processes enhance the usability of images for various purposes, from aesthetic improvements to privacy protection and the development of machine learning systems. The traditional approach, which relies on manual work and the skill of the operator, is inefficient and resource-intensive. My project aims to address these challenges by exploring the ability to identify and remove specific objects, such as vehicles, from digital images as a first step before expanding to other objects.

#### Project Description
My project consists of two main parts:
1. Object Detection in Images
2. Object Removal from Images

**Object Detection in Images**

1. **Data:**
   For this project, I chose to use the COCO-2017 dataset, known for its diversity of images containing objects in various contexts, making it ideal for training models capable of understanding and identifying objects in images. Additionally, I used the FiftyOne library for downloading and preprocessing the data.

   The dataset included 6,000 training images and 2,000 validation images. Due to computational constraints, I reduced the dataset to images containing only vehicles (cars, buses, trucks, motorcycles), and the library limited the validation set to 796 images.

   During the initial experiment phase, the process of downloading images and preprocessing them was done manually. However, I found that many images did not meet the research needs as they did not contain the relevant objects – vehicles, which were supposed to be the focus of the model's learning. Addressing this challenge led me to search for and explore more efficient methods for performing the task. In the research process, I discovered the FiftyOne library, which became a significant tool in my project.

   The FiftyOne library provides an advanced solution for downloading and preliminary processing of images by simplifying these processes significantly. It allows users to accurately define which types of objects should appear in the dataset, thus controlling the relevance and quality of the received data. Additionally, the library supports the ability to perform visualizations of the images, enabling me to evaluate the raw material before starting the model's prolonged training process.

2. **Preprocessing:**
   The preprocessing stage of the data in this project included two main steps:

   1. **Data Format Conversion:** The first task was to convert the dataset from the COCO format, based on JSON, to the YOLO format, which uses TXT files. Initially, this process was done manually, where the images were kept in their original format, each accompanied by a suitable TXT file detailing data about specific objects identified in the image. However, after advanced functionality was introduced in the FiftyOne package, which allows automatic and quick conversion to YOLOv5 format, I abandoned the manual method in favor of the automatic method, saving considerable time and effort.
   
   2. **Data Cleansing:** Following the initial training phase, unsatisfactory performance was partly due to the presence of irrelevant objects in the images, which were not part of the vehicle categories I focused on. These objects caused the model to focus on incorrect elements, thus impairing detection accuracy. To solve this problem, I went through the conversion process again, this time defining any object that was not a vehicle as class=other and then removing it, preventing their inclusion in the training data. This action helped focus the model's learning and improve the ability to accurately identify the relevant vehicles.

3. **Object Detection Model Training:**
   For this project, I chose YOLOv5, the fifth version of the You Only Look Once series, as the primary model for real-time object detection. It is one of the newest and most advanced technologies in the field, offering significant improvements in speed and detection accuracy compared to previous versions. YOLOv5's popularity among computer vision developers is explained by the following advantages:

   1. **Speed and Efficiency:** High-speed image processing while maintaining maximum accuracy makes YOLOv5 suitable for real-time applications and environments with limited computing resources.
   2. **High Accuracy:** The model can accurately identify objects even under complex conditions, such as images with object overlaps and complicated backgrounds.
   3. **Flexibility and Accessibility:** Easy to use and adaptable to a wide range of detection tasks, supporting custom training on specific datasets.
   4. **Wide Community Support:** Enjoys a large and active community of developers and data scientists, providing many information sources, tools, and additions.

**Customizing the Model for Vehicle Detection:**
I decided to train the YOLOv5 model on my specific dataset to optimize it for vehicle detection in images. Through customized training, the model developed a deep understanding of the unique characteristics of vehicles, increasing detection accuracy and reducing the likelihood of errors. The goal was to enhance the ability to correctly and quickly identify vehicles, regardless of various conditions or types of vehicles.

**Model Tuning:**
I started training the model with an image size of 640, a batch of 8, and 100 epochs. I achieved an average precision and recall of 65%.

Therefore, I began improving the image size to 720 and then to 1024, with a batch size of 16, and enabled early stopping to reduce runtime in case there was no improvement, reaching an average of 75%.

A crucial point is that my data was not balanced between classes: cars constituted 60% of the cases, with 20% for buses, 20% for trucks, and 20% for motorcycles, resulting in relatively low accuracy due to the primary focus on cars. However, overall, when I tested the model on new images, it performed perfectly as required.

In analyzing the results obtained from training both models, it was found that the YOLOv5 custom model showed superior performance in object detection tasks in images. The success of this model in accurately and efficiently identifying objects led me to decide to continue using it as the basis for the next advanced and important steps in the project.

**Model Performance:**
 
1. **Training and Validation Loss:** Box loss, object loss, and class loss consistently decreased during training, indicating model improvement and convergence. Validation loss was higher than training loss but also decreased over time, suggesting the model generalized well.
2. **Precision and Recall:** Precision and recall during training were stable with a slight upward trend, around 0.65 for precision and 0.60 for recall. For validation, there was an increase in precision and recall, with convergence around 0.675 and 0.625, respectively.
3. **Mean Average Precision (mAP):** The mAP metric improved consistently during training and validation, reaching approximately 0.50 in training and 0.45 in validation for mAP at an IoU threshold of 0.5 to 0.95.
4. **Recall-Confidence Curve:** The curve showed that for high confidence levels, recall for all classes was high, converging to a value of 0.83 at a confidence level of 0.000.
5. **Precision-Recall Curve:** The model showed high precision for the bus class with a value of 0.831, while the truck class showed the lowest precision with a value of 0.554. The average precision for all classes was 0.694 at a threshold of 0.5.
6. **Precision-Confidence Curve:** Precision improved as confidence level increased, reaching a perfect value of 1.00 at a confidence level of 0.987 for all classes.
7. **F1-Confidence Curve:** The highest F1 score for buses was 0.831, indicating consistent and accurate detection. For all classes, the optimal score was 0.69 at a confidence level of 0.258.

Overall, my model demonstrated good performance with particularly strong detection capabilities for specific classes like vehicles and buses. It showed flexibility to adapt to different confidence levels while maintaining high accuracy, given the time and budget constraints that limited training to 6,000 out of 120,000 images, surpassing the performance of pre-trained models.

#### Object Removal from Images

**Object Removal Model:**
After focusing on the customized YOLOv5 model, I proceeded with using Mask R-CNN, which allowed me to accurately identify the detected objects in the image and separate them from the background. Each identified object was given a segmentation mask describing its precise shape. At this stage, I utilized two important functions for result analysis: the `iou` function and the `percent_within` function.

The `iou` (Intersection over Union) function calculates the IOU between each pair of boxes, serving as a key tool for assessing detection accuracy by measuring overlap. This function allowed me to accurately measure the model's detection quality and compare different masks.

The `percent_within` function calculates the percentage of points within a specific bounding box, enabling me to assess how much of the object or mask is within the desired area. Using this function was crucial for analyzing the information obtained from the masks and understanding the model's ability to accurately focus on objects.

After obtaining accurate segmentation masks, the process continued with the use of DeepFill technology, designed to fill the spaces created after object removal from the image. Using Generative Adversarial Networks (GANs), DeepFill reconstructed the missing parts of the image convincingly and consistently, creating seamless results without a trace of the removed object. With the help of advanced technologies such as DeepFill and my precise YOLOv5 and Mask R-CNN models, I successfully achieved the project's goal – efficient and accurate removal of objects from images while maintaining high visual quality.

### Results and Achievements
1. Efficiently automated the process of identifying and removing vehicles from images.
2. Developed an accurate object detection
3. Successfully filled gaps created by object removal using DeepFill technology.

### Conclusion and Future Directions
My project demonstrates the potential of using advanced deep learning techniques for automating complex image editing tasks. In the future, My aim is to expand our research to include other object types and improve the efficiency and accuracy of our models. Additionally, I plan to explore real-time applications of our technology in various fields such as security, automotive, and healthcare.



![output_figure](https://github.com/user-attachments/assets/d8ba201b-4203-4051-9a82-48f30db172ca)

### Comparision between our custom-trained yolo model versus yolov5XL at cars-object detection
![image](https://github.com/user-attachments/assets/9041677b-bbb2-4b38-b139-bd80f2c2efa4)

We can clearly see how our model scores higher for all the cars types than the largest yolov5 model

### Comparision between our platform and the best online platforms
![image](https://github.com/user-attachments/assets/2853ea95-facb-4779-903b-9d5ec043cbb1)
![image](https://github.com/user-attachments/assets/5768ebc5-dc8a-4b5f-99ff-9b3d30865de4)


