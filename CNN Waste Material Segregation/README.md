# Waste Material Segregation Using Convolutional Neural Networks (CNN)

An image classification system built with TensorFlow and Keras designed to automate waste segregation. By categorizing waste into distinct physical categories, this project aims to support sorting automation facilities, maximize recycling efficiency, minimize environmental pollution, and promote sustainable waste lifecycle management.

## 📌 Project Objectives
* **Automated Classification:** Build and optimize a CNN model capable of classifying common waste materials accurately into their respective structural categories.
* **Recycling Optimization:** Streamline high-efficiency preprocessing streams for waste separation pipelines to decrease global landfill load.
* **Feature Extraction:** Understand structural image properties across variations in object configurations (e.g., distinguishing wrinkled paper from clear plastic structures).

---

## 📊 Dataset Specifications
The pipeline utilizes an image database containing **7,625 image samples** mapped across 7 unique objective waste designations:

| Category Index | Class Label | Sample Descriptions |
| :--- | :--- | :--- |
| 1 | 'Cardboard' | Shipping boxes, structural packaging components |
| 2 | 'Food_Waste'| Coffee grounds, organic matter, tea bags, food scraps |
| 3 | 'Glass' | Clear, green, and amber bottles or container shards |
| 4 | 'Metal' | Aluminum soda cans, tin items, structural hardware |
| 5 | 'Paper' | Office documents, dynamic printing media, magazines |
| 6 | 'Plastic' | PET water bottles, industrial storage containers |
| 7 | 'Other' | Compound synthetic materials, ambiguous refuse |

### Data Preprocessing Properties
* **Resolution Configurations:** Uniform structural resize applied across variants down to '128 x 128' pixels.
* **Color Channels:** Standard 3-Channel RGB color format.
* **Normalization Scale:** Min-Max scaling executed pixel-wise to map values inside the closed interval $[0.0, 1.0]$.

---

## 🛠️ Software Requirements

The codebase assumes access to a python execution ecosystem matching the recommended framework dependency versions below:

* **Core Language Environment:** 'Python >= 3.10'
* **Numerical & Storage Computations:** 'numpy == 1.26.4', 'pandas == 2.2.2'
* **Data Visualization:** 'matplotlib == 3.10.0', 'seaborn == 0.13.2'
* **Computer Vision Processing:** 'opencv-python', 'pillow == 11.1.0'
* **Modeling & Deep Learning Engine:** 'tensorflow == 2.18.0', 'keras == 3.8.0'
* **Machine Learning Analytics:** 'scikit-learn == 1.6.1'

---

## 🏗️ Pipeline Implementation Steps

### 1. Exploratory Layout Setup
Data discovery routines load standard files from remote structural compressed objects ('data.zip'), map local structural directory directories to uniform structural metadata structures using 'pandas.DataFrame' patterns, and visualize objective class balances using 'seaborn.countplot'.

### 2. Multi-Layer Image Transform
Dynamic loading via 'cv2.imread' handles programmatic matrix conversion to uniform RGB scaling layouts, matrix reshape methods normalize multidimensional structures, and structural pixel values are scaled down by division constants:
$$\mathbf{X}_{\text{normalized}} = \frac{\mathbf{X}_{\text{raw}}}{255.0}$$

### 3. Deep Neural Net Formulation
The architecture builds upon a 2D deep convolution stack layout using:
* Conv2D Feature Extractors paired with Rectified Linear Unit ('ReLU') parameter mappings.
* Batch Normalization paths to stabilize and accelerate model convergence speeds.
* Dropout regularizers set iteratively between $25\%$ and $50\%$ to prevent over-fitting.
* Dense Output Layer using a 7-way 'Softmax' structural mapping layer:
$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
