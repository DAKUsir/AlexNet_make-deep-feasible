# Glaucoma Detection Using AlexNet – Learning Project

This repository contains a deep learning pipeline for **automatic glaucoma detection** from retinal fundus images using a **custom AlexNet model** built in PyTorch. This project was developed as a **learning exercise** in convolutional neural networks (CNNs) and medical image analysis. The model combines multiple datasets to improve generalization, including **Fundus Scan Dataset, ORIGA, and ACRIMA**.

**Dataset (Kaggle):** [https://www.kaggle.com/datasets/chauhanadityacse/glaucoma-detection](https://www.kaggle.com/datasets/chauhanadityacse/glaucoma-detection)

---

## 🧠 Project Overview

Glaucoma is a progressive eye disease that can lead to irreversible vision loss if not detected early. Early detection is crucial to prevent permanent damage. This project aims to provide a **hands-on learning experience** for building and training CNNs for medical image classification.

---

## Quick links

* Kaggle dataset used: [https://www.kaggle.com/datasets/chauhanadityacse/glaucoma-detection](https://www.kaggle.com/datasets/chauhanadityacse/glaucoma-detection)

---

**Key Objectives of This Project:**

- Understand **data preprocessing and augmentation** for medical images.  
- Implement a **classic CNN architecture (AlexNet)** from scratch in PyTorch.  
- Explore **multi-dataset integration** for better model generalization.  
- Evaluate model performance using **accuracy, classification report, and confusion matrix**.  
- Visualize predictions on sample images to understand model behavior.  

---

## 📂 Dataset

The model leverages three publicly available datasets:

1. **Fundus Scan Dataset** – organized into `Glaucoma_Positive` and `Glaucoma_Negative` folders.  
2. **ORIGA Dataset** – contains a `.mat` file (`OrigaList.mat`) with image paths and glaucoma labels.  
3. **ACRIMA Dataset** – labeled images with `_g_` in the filename indicate glaucoma cases.  

> Ensure that all datasets are downloaded, extracted, and dataset paths are updated in the notebook.

**Data Preparation Highlights:**

- Images resized to 224×224 (AlexNet input size).  
- RGB images used (no grayscale conversion).  
- Data augmentation applied during training: random horizontal flips and rotations.  
- Train/validation split: 80% train / 20% validation.

---

## 🏗️ Model Architecture – AlexNet

The project uses a **custom AlexNet** implementation, adapted for smaller medical image datasets.

**Architecture Details:**

| Layer | Type | Output Shape | Notes |
|-------|------|--------------|-------|
| 1 | Conv2d | 64×224×224 | 11×11 kernel, stride 4, padding 2 |
| 2 | ReLU | 64×224×224 | Activation |
| 3 | MaxPool2d | 64×55×55 | 3×3 kernel, stride 2 |
| 4 | Conv2d | 192×55×55 | 5×5 kernel, padding 2 |
| 5 | ReLU | 192×55×55 | Activation |
| 6 | MaxPool2d | 192×27×27 | 3×3 kernel, stride 2 |
| 7 | Conv2d | 384×27×27 | 3×3 kernel, padding 1 |
| 8 | ReLU | 384×27×27 | Activation |
| 9 | Conv2d | 256×27×27 | 3×3 kernel, padding 1 |
| 10 | ReLU | 256×27×27 | Activation |
| 11 | Conv2d | 256×27×27 | 3×3 kernel, padding 1 |
| 12 | ReLU | 256×27×27 | Activation |
| 13 | MaxPool2d | 256×13×13 | 3×3 kernel, stride 2 |
| 14 | Flatten | 43264 | Prepare for FC layers |
| 15 | Linear | 4096 | Fully connected |
| 16 | ReLU | 4096 | Activation |
| 17 | Dropout | 4096 | p=0.5 |
| 18 | Linear | 4096 | Fully connected |
| 19 | ReLU | 4096 | Activation |
| 20 | Dropout | 4096 | p=0.5 |
| 21 | Linear | 2 | Output classes: Healthy / Glaucoma |

> This is a simplified AlexNet for **smaller datasets**. Dropout layers help reduce overfitting.  

---

## 🏋️ Training

**Training Configuration:**

- **Loss function:** CrossEntropyLoss  
- **Optimizer:** Adam (learning rate 0.001)  
- **Batch size:** 32  
- **Epochs:** 20–25 (adjustable)  
- **Device:** GPU recommended  
- **Data augmentation:** random flips and rotations  

**Training Highlights:**

- Monitored **train/validation loss and accuracy** for each epoch.  
- Early stopping can be implemented to prevent overfitting.  
- Model weights are saved as `alexnet_glaucoma.pth`.  

```python
# Example command to start training
python train.py
````

---

## 📊 Evaluation

* **Metrics computed:**

  * Overall accuracy
  * Glaucoma-only accuracy
  * Classification report (precision, recall, f1-score)
  * Confusion matrix

* **Visualization:**

  * Random sample predictions from validation set
  * Correct predictions shown in green, incorrect in red

* **Insights:**

  * Glaucoma-only accuracy is crucial since detecting positive cases is clinically important.
  * Visualization helps understand model errors and biases.

---

## 🖼️ Sample Predictions

* Sample predictions are displayed with predicted and true labels.
* Helps in **qualitative analysis** of the model performance.
* Useful for **learning purposes** to interpret CNN behavior on medical images.

---

## 💾 Saving & Loading the Model

* Save model weights:

```python
torch.save(model.state_dict(), "alexnet_glaucoma.pth")
```

* Load model for inference:

```python
model.load_state_dict(torch.load("alexnet_glaucoma.pth", map_location=device))
model.eval()
```

---

## 📌 Notes

* This is primarily a **learning project** to understand CNNs on medical images.
* Proper **GPU and sufficient memory** are recommended for training.
* AlexNet may be further improved with **transfer learning** or **larger datasets**.
* Dataset paths in the notebook must match your local setup or Kaggle environment.

---

## 📄 License

**MIT License** – free to use, modify, and distribute for educational or research purposes.

---

## 👨‍💻 Author

**Aditya Chauhan**

* CSE Student | AI Enthusiast | Kaggle Contributor
* Linkedin: https://www.linkedin.com/in/aditya-chauhan-async/

```



Do you want me to do that?
```
