# Brain Tumor Detection using Machine Learning

## Description
This project aims to train image detection models to augment and or enhance the diagnostic accuracy of medical professionals in the early detection of brain tumors.

Our team evaluated and refined serveral machine learning models: Vision Transformer, U-Net with DenseNet-121, and YOLO on a pre-labelled brain tumour MRI
scans dataset. 

We have included a brief summary of our results and evaluated the potential impact of this project for real-world applications in medical diagnostics. For detailed results and evaluation, kindly reach out to us.

## Table of Contents
- [1. Installation and Setup](#1-installation-and-setup)
- [2. Dataset](#2-dataset)
- [3. Models](#3-models)
- [4. Results and Evaluation](#4-results-and-evaluation)
- [5. Future Work](#5-future-work)
- [6. Licenses](#6-licenses)
- [7. Acknowledgements](#7-acknowledgements)

## 1. Installation and Setup
Our project was run on Google Colab using Python 3. In addition, we utilized the following libraries, packages and dependencies for our analysis and model training. Kindly observe the recommended steps and requirements below, in order to replicate our environment and run our project.

### 1.1 Google Colab and Google Drive
Our modelling was performed in Google Colab and ingests datasets which were stored in Google Drive. When using our codes, kindly download the dataset into your Google Drive and mount or replace with your corresponding working directories. See also [section 2.2](#22-augmented-datasets).

### 1.2 Python Version
Our codes are compatible with Python 3.6 and above. Google Colab typically provides the latest Python 3 version. 

### 1.3 Required Libraries and Packages
Google Colab typically comes with many essential packages pre-installed, but some may need to be installed or updated depending on their usage in the project. Below is a detailed list of the libraries we used, categorized by their purpose:

#### 1.3.1 General Purpose Libraries
- `os`: For interacting with the operating system.
- `sys`: For accessing system-specific parameters and functions.
- `glob`: For retrieving files/pathnames matching a specified pattern.
- `random`: For generating random numbers.
- `re`: For regular expression matching operations.
- `yaml`: For YAML file parsing.
- `logging`: For tracking events that occur during runtime.
- `tempfile`: For generating temporary files and directories.

#### 1.3.2 Data Manipulation Libraries
- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.

#### 1.3.3 Data Visualization Libraries
- `matplotlib`: For creating static, interactive, and animated visualizations.
- `seaborn`: For statistical data visualization.
- `IPython.display`: For displaying objects in Python notebooks.

#### 1.3.4 Image Processing and Computer Vision Libraries
- `Pillow` (PIL): For manipulating different image file formats.
- `cv2` (OpenCV): For real-time computer vision.

#### 1.3.5 Machine Learning and Deep Learning Libraries
- `torch`: For tensor operations and building neural networks.
- `torchvision`: For image processing tools and dataset loaders.
- `torchmetrics`: For computing metrics for PyTorch.
- `pytorch_lightning`: For organizing PyTorch code, simplifying complex model training.
- `torch.utils.tensorboard`: For logging and visualizing data during training.
- `ultralytics`: For utilizing advanced machine learning models, specifically the **YOLO** model.
- `monai`: For deep learning in healthcare imaging, providing tools and pre-trained models, specifically the **U-Net** and **DenseNet121** models.
- `timm`: For pre-trained deep learning models in PyTorch, specifically the **Vision Transformer** model

### 1.4 Installation Commands
To install the necessary libraries directly into Google Colab, you may use the following pip commands:

```bash
!pip install numpy pandas seaborn matplotlib opencv-python-headless Pillow PyYAML ultralytics
!pip install torch torchvision torchmetrics pytorch_lightning monai timm
```


## 2. Dataset
Our project utilizes a pre-labelled MRI brain tumor dataset from [Roboflow Universe](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset). 

The original dataset comprises 2,443 jpg images with a 640 x 640 resolution. Accompanying each image file is a text label file that details the tumor type and bounding box coordinates. Each pair of image and label files share the same name with a prefix indicating the tumor type as follows:
- *Tr-pi*: representing Pituitary tumors (class 3 label) 
- *Tr-me*: representing Meningioma tumors (class 1 label)
- *Tr-gl*: representing Glioma tumors (class 0 label)
- *Tr-no*: representing cases with No Tumor (class 2 label)

The dataset was split into 1,695 training, 502 validation, and 246 test. 

### 2.1 Data Preprocessing
To enhance our model training process, we performed geometric transformations on the original dataset to double the total number of training images to 3,390 and validation images to 1,004 and preserved the same class balances; no augmentation was done for test data. 

The following transformation techniques were used:
- `Rotate`: Random rotations between -10° and 10°, to mimic head tilting and rotation.
- `ShiftScaleRotate`: A combination of shifting and scaling (without rotate), for robustness against positional and size variations.
- `HorizontalFlip`: Flips the image horizontally (left to right).
- `ElasticTransform`: Elastic transformations, introducing non-linear deformations for mimicking real-world scanning inconsistencies.

### 2.2 Augmented Datasets
To run our codes successfully, kindly use our augmented datasets:
- [Train dataset](https://drive.google.com/file/d/1Ig4FwFcuzyBOhBFz3Av3DSEsFtFV13yb/view?usp=sharing)
- [Valid dataset](https://drive.google.com/file/d/1FxnLwNuHN1birGh24sTV5eR4bIRS75rx/view?usp=sharing)
- [Test dataset](https://drive.google.com/file/d/1Ru6gLRWap6_s2wWCrBTmIEvDrnjKbhGo/view?usp=drive_link)


## 3. Models
We utilized three models for our analysis:
- [Vision Transformer (ViT)](https://github.com/gabigarms/BT5153-Final-Project/blob/main/codes/vit.ipynb)
- [U-Net with DenseNet121](https://github.com/gabigarms/BT5153-Final-Project/blob/main/codes/unet_densenet.ipynb)
- [YOLO](insert github link)

These models were selected based on their abilities to identify and adaptively learn critical details and features. These models also have diverse real world use cases. For example, VIT is prominent in the field of computer vision for its image classification abilities and has been adopted in fields with large datasets e.g. social media and e-commerce. U-Net was original designed and is widely adopted for medical imaging segmentation. YOLO has been widely adapted for traffic monitoring due to its speed and efficiency in real-time object detection.  


## 4. Results and Evaluation
*For detailed results and evaluation, kindly reach out to us*

To ensure a consistent and comprehensive analysis of all three models, we adopted three standard metrics for evaluation:
- **Precision**: Measures the accuracy of the tumor predictions.
- **Recall**: Indicates how well the model identifies actual tumors.
- **F1 Score**: Provides a balanced measure of precision and recall, serving as a single metric for overall model performance.

### 4.1 Baseline and Fine-tuning
All models were trained and finetuned using strategies tailored specifically for each. You may view the baseline and fine-tuned results for each model in our [images folder](https://github.com/gabigarms/BT5153-Final-Project/tree/main/images). Please reach out to us for detailed information on our modelling and fine-tuning methodologies.

### 4.2 Consolidated Results
The consolidated best results from each model suggest the following:
![Consolidated Results](https://raw.githubusercontent.com/gabigarms/BT5153-Final-Project/main/images/consolidated_results.jpg "Consolidated Results after Fine-tuning")

- **ViT:** Achieved an F1 score of 0.823, and is ideal for complex diagnostic tasks requiring detailed image analysis in research-driven or large-scale applications. However, it demands significant computational resources.
- **U-Net with DenseNet-121:** Scored a mean F1 of 0.902, excelling in precise image segmentation and classification. This model is suitable for structured diagnostic environments but like ViT, requires considerable computational resources (*we used the simple mean of U-Net and DenseNet-121's F1 scores*). 
- **YOLOv8n:** With an F1 score of 0.906, this model stands out in clinical scenarios where speed is critical. It offers rapid diagnosis capabilities, making it essential in emergency settings, although it may trade off some precision for speed..

### 4.3 Summary Recommendations 
No single model is universally the best, rather each model is and can be tailored to certain specific needs and scenarios e.g. ViT can be used in medical research for its depth and detail in resource-abundant settings; U-Net and DenseNet-121 for accuracy in structured medical imaging workflows; and YOLOv8n for speed in medical emergency imaging workflows.

The main challenge across all models is their substantial computational demands and operational costs, which could be prohibitive in resource-limited settings.


## 5. Future Work
This project was constrained largely in terms of computational resources which allowed us to only perform shallow-depth training and basic fine-tuning.

Future contributors can also consider expanding the dataset through additional labelled data or further augmentation.

Nonetheless this project is still a valuable and reliable foundation for future modifications, enhancements and or other contributions.


## 6. Licenses
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

The MIT License allows you to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided that the above copyright notice and this permission notice are included in all copies or substantial portions of the software.


## 7. Acknowledgements
Once again we would like to express our deepest gratitude to Roboflow user *Ali Rostami* for the well labelled tumor dataset in [Roboflow Universe](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset). 
