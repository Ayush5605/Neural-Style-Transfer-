# 🎨 Neural Style Transfer

A deep learning-based **Neural Style Transfer** application that combines the content of one image with the artistic style of another image using **Adaptive Instance Normalization (AdaIN)**.

The project uses a pretrained **VGG encoder** to extract image features and a trained **decoder** to reconstruct the stylized image. The trained models are hosted on Hugging Face, while the application is deployed as an interactive Gradio application using Hugging Face ZeroGPU.

## 🚀 Live Demo

**Hugging Face Space:**  
https://huggingface.co/spaces/Ayush5605/NST

**Trained Model:**  
https://huggingface.co/Ayush5605/NST_Model

> The demo uses Hugging Face ZeroGPU, so GPU resources are allocated dynamically when an inference request is made.

---

## ✨ Features

- 🖼️ Upload a content image
- 🎨 Upload a style image
- 🎚️ Control style strength using the Alpha parameter
- ⚡ GPU-accelerated inference using Hugging Face ZeroGPU
- 🧠 AdaIN-based neural style transfer
- 📥 Download the generated stylized image
- 🤗 Models hosted on Hugging Face Hub
- 🌐 Interactive Gradio web interface

---

## 🧠 How It Works

The application follows an encoder → AdaIN → decoder pipeline.

```text
             Content Image
                   │
                   ▼
             VGG Encoder
                   │
                   │ Content Features
                   │
                   ├──────────────┐
                   │              │
                   │              ▼
                   │        AdaIN Operation
                   │              ▲
                   │              │
                   │       Style Features
                   │              │
             Style Image          │
                   │              │
                   ▼              │
             VGG Encoder ─────────┘
                                  │
                                  ▼
                           Alpha Blending
                                  │
                                  ▼
                               Decoder
                                  │
                                  ▼
                         Stylized Image
```

### 1. Feature Extraction

Both the content and style images are passed through a pretrained VGG encoder.

The encoder extracts high-level feature representations rather than directly working with raw pixels.

### 2. Adaptive Instance Normalization

AdaIN aligns the channel-wise mean and variance of the content feature representation with those of the style representation.

This transfers the statistical characteristics of the style image into the content features.

### 3. Alpha Blending

The `alpha` parameter controls how strongly the style is applied:

```text
alpha = 0.0 → mostly original content
alpha = 1.0 → maximum style transfer
```

### 4. Image Reconstruction

The transformed feature representation is passed through the trained decoder to reconstruct the final stylized image.

---

## 🛠️ Tech Stack

### Deep Learning
- Python
- PyTorch
- Torchvision
- VGG
- Adaptive Instance Normalization (AdaIN)

### Web Application
- Gradio
- Hugging Face Spaces
- Hugging Face ZeroGPU

### Model Hosting
- Hugging Face Hub

### Image Processing
- Pillow

---

## 📁 Project Structure

```text
Neural Style Transfer/
│
├── Hugging_face_Space/
│   ├── app.py
│   ├── README.md
│   ├── requirements.txt
│   │
│   └── utils/
│       ├── models.py
│       └── utils.py
│
├── NST_Code/
│   ├── train.py
│   ├── resize_dataset.py
│   ├── app.py
│   ├── requirements.txt
│   └── utils/
│       ├── models.py
│       └── utils.py
│
├── .gitignore
└── README.md
```

> The exact local project structure may contain additional training and dataset files that are excluded from deployment.

---

## 📦 Models

The trained model files are hosted separately on Hugging Face:

**Repository:**  
https://huggingface.co/Ayush5605/NST_Model

Current inference models:

```text
vgg_normalised.pth
decoder_80.pth
```

The Gradio application downloads these files automatically using `hf_hub_download()`.

The model weights are therefore not required to be committed to the application repository.

---

## 💻 Running Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Neural-Style-Transfer
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

For the Gradio application:

```bash
pip install -r Hugging_face_Space/requirements.txt
```

### 4. Run the Gradio application

```bash
cd Hugging_face_Space
python app.py
```

The application will provide a local Gradio URL.

> The deployed version is configured for Hugging Face ZeroGPU and expects CUDA-enabled inference.

---

## 🎯 Using the Application

1. Open the live demo.
2. Upload a **Content Image**.
3. Upload a **Style Image**.
4. Adjust **Style Strength**.
5. Click **Generate Stylized Image**.
6. Download the generated result.

For initial experiments, try an Alpha value between:

```text
0.7 – 1.0
```

---

## 🔬 Training

The decoder was trained using the project's training pipeline.

The training code is kept separate from the deployed inference application.

```text
Training
   │
   ├── Content Dataset
   ├── Style Dataset
   ├── VGG Encoder
   └── Decoder Training
            │
            ▼
      Trained Decoder
            │
            ▼
       Hugging Face
            │
            ▼
      Gradio Inference
```

The deployment does **not** retrain the model. It only performs inference using the pretrained encoder and trained decoder.

---

## ⚡ Deployment

The application is deployed using:

```text
Gradio
   ↓
Hugging Face Spaces
   ↓
ZeroGPU
   ↓
PyTorch GPU Inference
```

ZeroGPU dynamically allocates GPU resources when the style-transfer function is executed.

---

## 📌 Future Improvements

Potential improvements include:

- Support for higher-resolution output
- Additional trained decoder checkpoints
- Batch image processing
- Image history/gallery
- More style controls
- Preset artistic styles
- Improved preprocessing and postprocessing
- Performance optimization for larger images

---

## 👨‍💻 Author

**Ayush Auti**

Computer Engineering Student  
Pune, India

---

## ⭐ Acknowledgements

This project is based on the concept of **Arbitrary Style Transfer using Adaptive Instance Normalization (AdaIN)** and uses VGG-based feature extraction with a trained decoder.

If you find the project useful, consider giving the repository a ⭐.
