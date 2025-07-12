# 🖼️ Image Classification using CNN (CIFAR-10 Dataset)

## 📌 Project Overview

This deep learning project demonstrates how to classify images from the **CIFAR-10 dataset** using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**.  
The model is trained to recognize 10 object categories like airplanes, cars, animals, and ships.

---

## 🧾 Dataset Information

- 🗂 Dataset: `CIFAR-10` (from `tensorflow.keras.datasets`)
- 👨‍🏫 Classes:
  - `airplane`, `automobile`, `bird`, `cat`, `deer`,  
    `dog`, `frog`, `horse`, `ship`, `truck`
- 🖼️ Image Shape: 32x32 pixels with 3 color channels (RGB)  
- 📊 Total samples:
  - `x_train`: 50,000 images  
  - `x_test`: 10,000 images

---

## 🧹 Data Preprocessing

1. 📥 Loaded data using `datasets.cifar10.load_data()`  
2. ✅ Normalized pixel values between 0 and 1:
   ```python
   x_train, x_test = x_train / 255.0, x_test / 255.0
   ```
3. 🧾 Class labels mapped using `class_names` list

---

## 🧠 Model Architecture

Constructed a **Sequential CNN** model with the following layers:

```python
Conv2D(32, (3,3), activation='relu') ➡️ MaxPooling2D(2,2)
Conv2D(64, (3,3), activation='relu') ➡️ MaxPooling2D(2,2)
Conv2D(64, (3,3), activation='relu')
Flatten ➡️ Dense(64, activation='relu') ➡️ Dense(10)
```

- 🔧 Compiled with:
  - Optimizer: `adam`  
  - Loss: `sparse_categorical_crossentropy`  
  - Metrics: `accuracy`

---

## 🏋️ Model Training

- ✅ Trained for **20 epochs** with **10% validation split**  
- 📈 Plotted:
  - Accuracy vs Epochs  
  - Loss vs Epochs  

---

## 📈 Evaluation

- 🤖 Model evaluated on the test set using:
  ```python
  accuracy_score(y_test, pred)
  ```
- 🔢 Predictions extracted using:
  ```python
  pred = np.argmax(y_pred, axis=1)
  ```
- 📊 Sample prediction visualized using `matplotlib`

---

## 💾 Model Saving

- 🧠 Saved the trained model as:
  ```python
  model.hdf5
  ```

---

## 📌 Key Observations

- CNNs are highly effective in handling image classification tasks  
- CIFAR-10 provides a challenging, yet manageable image dataset for beginners  
- Visualization helps track performance and identify underfitting/overfitting  
- Accuracy improves significantly by using convolutional layers and pooling

---

## 🧰 Libraries Used

- `tensorflow` 🧠  
- `matplotlib.pyplot` 📊  
- `numpy` 🔢  
- `sklearn.metrics` 📈

---

## 🚀 Future Improvements

- 🧪 Add **data augmentation** to increase generalization  
- 🌊 Use **Batch Normalization** or **Dropout** layers to prevent overfitting  
- ⚡ Try **ResNet**, **VGG**, or **MobileNet** architectures  
- 📲 Convert to **TFLite** for mobile deployment

---

## 🙏 Acknowledgments

- CIFAR-10 dataset provided by the Canadian Institute for Advanced Research  
- Inspired by foundational deep learning projects for computer vision

---

## 📜 License

This project is licensed under the **MIT License** ✅
