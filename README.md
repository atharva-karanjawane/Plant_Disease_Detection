# Crop Disease Prediction System

![Crop Disease Prediction](docs/images/banner.png)

A deep learning-based system for automatic crop disease classification using RGB images. This project uses EfficientNet-B0 to classify crop diseases from standard RGB images, enabling rapid, accurate, and scalable disease detection for farmers and agronomists.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [API](#api)
  - [Mobile Deployment](#mobile-deployment)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 🌟 Overview

Early detection of crop diseases is crucial for sustainable agriculture and food security. This project provides an automated solution for identifying crop diseases using only RGB images, making it accessible to farmers worldwide without requiring specialized equipment.

The system can identify 38 different crop-disease combinations across 14 crop species with high accuracy (94.2% on test set), providing immediate feedback and treatment recommendations.

## ✨ Features

- **High Accuracy Disease Classification**: 94.2% accuracy on test set
- **Explainable Predictions**: Grad-CAM visualizations show which parts of the image influenced the prediction
- **Mobile-Ready Deployment**: TensorFlow Lite model for on-device inference
- **Continuous Learning Capabilities**: System can be retrained with new data
- **User-Friendly Web Interface**: Easy-to-use interface for uploading and analyzing images
- **Treatment Recommendations**: Provides actionable advice based on detected diseases
- **RESTful API**: Integrate disease detection into other agricultural systems
- **Domain-Specific Data Augmentation**: Custom augmentation techniques for crop disease images


## 🔧 Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crop-disease-prediction.git
   cd crop-disease-prediction