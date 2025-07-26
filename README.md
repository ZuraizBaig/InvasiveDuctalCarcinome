# Breast Cancer Classifier

This project implements a convolutional neural network (CNN) to detect **Invasive Ductal Carcinoma (IDC)** from breast cancer histopathology images.

## Features

- Preprocessing and augmentation of image data
- CNN architecture built with TensorFlow/Keras
- Training and evaluation of the model
- Visualizations of performance

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebook:
   Open `Invasive_Ductal_Carcinoma_(Breast_Cancer).ipynb` in Jupyter or Colab.

3. Or run the script:
   ```bash
   python Breast_Cancer_Classifier.py
   ```

## Data

Place your breast histopathology images dataset in the `data/` folder. Structure should follow:
```
data/
  └── class1/
  └── class2/
```

## Output

Model checkpoints and training results can be saved in the `model/` folder.

---
