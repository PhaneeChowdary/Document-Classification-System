# Mortgage Document Classifier

A deep learning-based document classification system that automatically categorizes mortgage-related documents using Vision Transformers (ViT) and OCR capabilities.

## Features

- Multi-label document classification for 5 mortgage document types:
  - Forms
  - Approval Letters
  - Payment Records/Invoices
  - Financial Statements/Budgets
  - Property Specifications

- Built with state-of-the-art components:
  - Vision Transformer (ViT-B-16) for image classification
  - Tesseract OCR for text extraction
  - MLflow for experiment tracking
  - PyTorch for deep learning


## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Tesseract OCR:
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

## Usage

### Training

Run the training script:

```bash
python app.py
```

Training features:
- Automatic MLflow experiment tracking
- Progress bars with live metrics
- Checkpoint saving for best models
- MPS/CPU support
- Multi-processing optimizations

## Model Architecture

- Base model: ViT-B-16 pretrained on ImageNet
- Custom head:
  - Linear(768, 256)
  - ReLU
  - Dropout(0.5)
  - Linear(256, 5)
  - Sigmoid activation

### Thank you