# 🫁 Tuberculosis Detection Using Keras & PyTorch with LIME and Grad-CAM

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-powered chest X-ray screening system for Tuberculosis detection with visual explainability.**

[Live Demo](#) · [Report Bug](#) · [Request Feature](#)

</div>

---

## 📋 Table of Contents

- [Goal](#-goal)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Explainability](#-explainability)
- [Results](#-results)

---

## 🎯 Goal

The goal of this project is to build a machine learning model using **Keras** for predicting the presence of **Tuberculosis (TB)** from chest X-ray images. Additionally, the project provides model explainability using:

- **LIME** *(Local Interpretable Model-agnostic Explanations)* — highlights image regions influencing predictions
- **Grad-CAM** *(Gradient-weighted Class Activation Mapping)* — overlays heatmaps on X-rays to show model focus areas

This combination ensures the model is not only accurate but also **interpretable and trustworthy** for real-world medical use.

---

## 🖥️ Demo

<div align="center">

| Original X-Ray | Grad-CAM Heatmap | Prediction |
|:-:|:-:|:-:|
| ![xray](#) | ![gradcam](#) | ✅ Normal / ⚠️ TB Positive |

</div>

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Deep Learning** | TensorFlow · Keras · PyTorch |
| **Explainability** | LIME · Grad-CAM · SHAP |
| **Web Backend** | Flask · Flask-CORS |
| **Image Processing** | OpenCV · Pillow |
| **Data & Visualization** | NumPy · Pandas · Matplotlib · Seaborn |
| **Frontend** | HTML5 · CSS3 · JavaScript |

---

## 📁 Project Structure

```
Tuberculosis-Detection-Model/
│
├── tb_detection_app/
│   ├── app.py                  # Flask API backend
│   ├── tb_model.h5             # Trained Keras model
│   ├── requirements.txt        # Python dependencies
│   └── static/
│       └── index.html          # Frontend UI
│
├── TB_Chest_Radiography_Database/
│   ├── Normal/                 # Normal chest X-rays
│   └── Tuberculosis/           # TB-positive chest X-rays
│
├── notebook.ipynb              # Training + LIME + Grad-CAM notebook
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/ShivankSubanshi/Tuberculosis-Detection-Model.git
cd Tuberculosis-Detection-Model
```

**2. Install dependencies**
```bash
cd tb_detection_app
pip install -r requirements.txt
```

**3. Add your trained model**

Run the notebook to train and then save your model:
```python
lenet_model.save("tb_detection_app/tb_model.h5")
```

---

## 🚀 Usage

**Start the Flask server**
```bash
cd tb_detection_app
python app.py
```

**Open in browser**
```
http://localhost:5000
```

Then upload any chest X-ray (JPG/PNG) and click **Analyze** to get:
- ✅ **Prediction** — Normal or TB Positive
- 📊 **Confidence Score** — model certainty percentage
- 🔥 **Grad-CAM Overlay** — heatmap showing regions of interest

---

## 🧠 Model Architecture

A custom **LeNet-based CNN** is built using Keras:

```
Input (256×256×3)
    → Conv2D(6, 5×5, ReLU)  → MaxPooling(2×2)
    → Conv2D(16, 5×5, ReLU) → MaxPooling(2×2)
    → Flatten
    → Dense(120, ReLU) → Dense(84, ReLU)
    → Dense(1, Sigmoid)   ← Binary output
```

**Training details:**
- Optimizer: `Adam`
- Loss: `Binary Cross-Entropy`
- Epochs: `12`
- Batch Size: `32`
- Regularization: Early Stopping · Learning Rate Scheduling

---

## 🔍 Explainability

### Grad-CAM
Grad-CAM computes gradients of the predicted class score with respect to the last convolutional layer's feature maps, producing a **heatmap** that highlights the most discriminative lung regions.

### LIME
LIME perturbs the input image and observes how predictions change, generating a **superpixel mask** that identifies which regions most contributed to the classification decision.

---

## 📊 Results

| Metric | Score |
|---|---|
| **Test Accuracy** | ~XX% |
| **Precision** | ~XX% |
| **Recall** | ~XX% |
| **F1-Score** | ~XX% |

> 📝 Fill in your actual evaluation scores from the notebook's classification report.

**Confusion Matrix and ROC Curve** are available in the notebook.

---

## ⚕️ Disclaimer

> This tool is intended as a **screening aid only**. All results must be reviewed and confirmed by a qualified radiologist or physician before any clinical decision is made. Do not use as a sole diagnostic resource.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
  Made with ❤️ | TB Detection Research Project · 2026
</div>
