# 🌲 Forest Fire Detection using Deep Learning and Machine Learning

This project focuses on detecting **forest fires** from images using both **Convolutional Neural Networks (CNNs)** and **classical machine learning models**.  
By comparing deep learning with traditional approaches such as **SVM**, **Logistic Regression**, and **KNN**, the project identifies the most effective method for accurate forest fire detection.

---

## 🧠 Project Overview

Wildfires are among the most destructive natural disasters, causing severe environmental and economic damage.  
This project demonstrates how **AI-driven image classification** can assist in **early fire detection**, potentially reducing disaster impact.

The notebook trains and compares several models:
- **CNN (Convolutional Neural Network)**
- **Support Vector Classifier (SVC)**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**

Each model is evaluated using accuracy, precision, recall, and F1-score metrics.

---

## ⚙️ Technologies Used

**Language:** Python  
**Main Libraries:**
- TensorFlow & Keras – Deep learning (CNN model)
- Scikit-learn – Classical ML models (SVC, Logistic Regression, KNN)
- NumPy – Numerical operations
- Matplotlib – Visualization
- Pillow (PIL) – Image preprocessing
- Google Colab – Execution environment (with Google Drive integration)

---

## 🗂️ Dataset

The dataset used is a **custom image dataset** named **“ForestFireDataset”**, consisting of **two classes**:
- 🔥 `fire` – Images containing forest fire  
- 🌳 `no_fire` – Images without fire  

| Split | Images | Classes |
|--------|----------|----------|
| Training | 1520 | Fire / No Fire |
| Testing | 380 | Fire / No Fire |

Data is automatically augmented using `ImageDataGenerator` to improve model generalization.

---

## 🧩 Model Training and Evaluation

### Models Implemented
| Model | Type | Accuracy |
|--------|------|-----------|
| **CNN** | Deep Learning | **~96%** |
| **SVC** | Classical ML | **~94%** |
| **Logistic Regression** | Classical ML | **~93%** |
| **KNN** | Classical ML | **~85%** |

### Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Example visualizations include:
- Comparison of model accuracies  
- CNN training accuracy over epochs  
- Precision and F1-score per class  

---

## 📊 Results Summary

**CNN outperformed all other models**, achieving:
- **Accuracy:** ~96%  
- **F1-Score:** ~0.96  
- **Precision:** ~0.95  

The results confirm that **deep learning models** are superior for complex, image-based classification tasks compared to traditional ML models.

---

## ▶️ How to Run the Project

This notebook is designed to run on **Google Colab**.

### Steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Forest_Fire_Detection.git
   cd Forest_Fire_Detection
   ```

2. **Open the notebook in Google Colab**
   - Upload `Forest_Fire_Detection.ipynb` to your Google Drive.
   - Open it with Google Colab.

3. **Mount Google Drive**
   Inside the notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Update Dataset Paths**
   Replace the dataset paths with your own:
   ```python
   train_dir = '/content/drive/MyDrive/ForestFireDataset/train'
   test_dir = '/content/drive/MyDrive/ForestFireDataset/test'
   ```

5. **Run all cells sequentially**
   The notebook will train each model and visualize the comparison metrics.

---

## 📈 Example Visualization

```
Comparison of Model Accuracies:
CNN: 0.96
SVC: 0.94
Logistic Regression: 0.93
KNN: 0.85
```

CNN training accuracy also shows near-perfect learning stability over 10 epochs.

---

## 🔮 Future Improvements

- Implement **transfer learning** using pretrained models (e.g., MobileNet, ResNet).  
- Enable **real-time video stream detection** via OpenCV.  
- Deploy the model on **edge devices** with TensorFlow Lite.  
- Integrate **cloud-based alerts** for detected fire regions.

---

## 📜 License

This project is released under the **MIT License**.  
You’re free to use, modify, and distribute this code with attribution.

---

## 🙌 Acknowledgments

- TensorFlow/Keras documentation  
- Scikit-learn community  
- Google Colab for providing GPU compute  
- Dataset contributors of *ForestFireDataset*  
