# Historical Image Dating AI

A deep learning framework for estimating the historical period of photographs and cultural artifacts using Computer Vision.

---

## Features

- Custom CNN
- ResNet50 Transfer Learning
- Support Vector Machine baseline
- Random Forest baseline
- TensorBoard support
- Grad-CAM visualization
- ONNX Export
- Docker support
- GitHub Actions CI
- Automatic evaluation
- Confusion matrices
- Precision / Recall / F1

---

## Dataset Structure

data/

    train/

        1800_1850/

        1851_1900/

        1901_1950/

        1951_2000/

    validation/

    test/

Each folder should contain JPG or PNG images.

---

## Installation

```bash
git clone https://github.com/username/historical-image-dating-ai

cd historical-image-dating-ai

pip install -r requirements.txt
```

---

## Train CNN

```bash
python train.py
```

---

## Predict

```bash
python predict.py image.jpg
```

---

## TensorBoard

```bash
tensorboard --logdir logs
```

---

## Export ONNX

```bash
python export_onnx.py
```

---

## Models

| Model | Purpose |
|---------|----------|
| CNN | Baseline Deep Learning |
| ResNet50 | Transfer Learning |
| SVM | Traditional ML |
| Random Forest | Traditional ML |

---

## Metrics

- Accuracy

- Precision

- Recall

- F1 Score

- Confusion Matrix

---

## Future Work

- Vision Transformers

- CLIP

- Multi-modal metadata

- Self-supervised learning

- OCR integration

- Geographic metadata

- Ensemble learning
