# Eco-Vision: AI-Powered Satellite Image Classifier

Eco-Vision is a Deep Learning application designed for Land Use and Land Cover (LULC) classification. It leverages Computer Vision to analyze satellite imagery and categorize land types into 10 distinct classes with high precision.

## 🚀 Project Overview
This project addresses the challenge of automated environmental monitoring by processing Sentinel-2 satellite data. The core engine is built using **Transfer Learning** on a pre-trained **MobileNetV2** architecture, optimized specifically for the EuroSAT dataset.

### 🎯 Key Features
- **High-Accuracy Classification:** Achieved an 87.5% validation accuracy using state-of-the-art Deep Learning techniques.
- **Optimized Pipeline:** Implemented custom **Rescaling layers** and **Early Stopping** to ensure robust model performance and prevent overfitting.
- **Real-time Inference:** Integrated with a **Streamlit** web interface for instant image-to-prediction results.
- **Scalable Architecture:** Designed to handle diverse geographical features including Forests, Industrial zones, Residential areas, and more.

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Deep Learning:** TensorFlow 2.x, Keras
- **Web Framework:** Streamlit
- **Data Manipulation:** NumPy, Pillow (PIL)
- **Version Control:** Git & GitHub

## 📊 Dataset: EuroSAT
The model is trained on the **EuroSAT RGB dataset**, consisting of 27,000 labeled satellite images. It classifies land cover into 10 categories:
`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`.

## ⚙️ Installation and Usage

**1. Clone the repository:**
```bash
git clone [https://github.com/subash2000-eng/EcoVision-AI-Satellite-Classifier.git](https://github.com/subash2000-eng/EcoVision-AI-Satellite-Classifier.git)
cd EcoVision-AI-Satellite-Classifier

2. Install dependencies:

pip install -r requirements.txt

3. Run the application:

streamlit run app.py


🧠 Technical Highlights
Transfer Learning: Utilized MobileNetV2's feature extraction capabilities to work efficiently with low-resolution (64x64) satellite data.

Domain Adaptation: Successfully handled domain shift challenges by implementing specific preprocessing layers to match the input distribution of pre-trained weights.

Deployment: Focused on creating a seamless end-to-end user experience from image upload to final classification output.

Developed as part of a Deep Learning and Remote Sensing exploration project.