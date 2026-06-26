
# 📷 Webcam Digit Recognizer

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A real-time **Handwritten Digit Recognition** application built with **Python**, **TensorFlow/Keras**, **OpenCV**, and **NumPy**.

The application uses a Convolutional Neural Network (CNN) trained on the **MNIST** dataset to recognize handwritten digits (0–9) from a live webcam feed.

---

## ✨ Features

* 📷 Real-time webcam digit recognition
* 🧠 CNN model trained on the MNIST dataset
* 🎯 Automatic digit detection using contour extraction
* 📦 Bounding box around detected digit
* 📊 Confidence score display
* 🔝 Top-3 prediction probabilities
* 🖼️ 28×28 processed image preview
* ⚡ Fast and lightweight

-

##  Requirements

* Python 3.9 or later
* Webcam
* TensorFlow
* OpenCV
* NumPy

Install the required packages:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

or

```bash
pip install -r requirements.txt
```

---

##  Train the Model

If you don't already have **mnist_cnn.h5**, train the model first:

```bash
python train_mnist.py
```

The script will:

* Download the MNIST dataset
* Train the CNN model
* Save the trained model as:

```text
mnist_cnn.h5
```

---

##  Run the Application

Start the webcam recognizer:

```bash
python webcam.py
```

A webcam window will open with a green box.

1. Hold a handwritten digit inside the green box.
2. Keep the digit centered and steady.
3. The model predicts the digit in real time.
4. Press **Q** to quit.

---

##  How It Works

1. Capture live video from the webcam.
2. Extract the Region of Interest (ROI).
3. Convert the ROI to grayscale.
4. Apply Gaussian Blur to reduce noise.
5. Perform adaptive thresholding.
6. Detect the largest contour.
7. Crop and resize the digit while preserving its aspect ratio.
8. Center the digit on a 28×28 canvas.
9. Normalize pixel values.
10. Feed the processed image into the CNN model.
11. Display:

* Predicted digit
* Confidence score
* Top-3 predictions
* Processed 28×28 preview

---

## 💡 Tips for Better Accuracy

* Write the digit **large and bold**.
* Use a **black marker** on **white paper**.
* Keep the paper **flat**.
* Ensure **good lighting**.
* Avoid shadows and glare.
* Keep the digit centered inside the green box.

---

##  Example Output

```text
Prediction : 7
Confidence : 99.63%

Top Predictions

7 : 99.63%
1 : 0.28%
9 : 0.06%
```

---

## ❗ Troubleshooting

| Problem              | Solution                                                                  |
| -------------------- | ------------------------------------------------------------------------- |
| Failed to load model | Make sure `mnist_cnn.h5` is in the project folder.                        |
| Webcam not detected  | Check your webcam connection and ensure no other application is using it. |
| No prediction        | Improve lighting or move the digit fully inside the green box.            |
| Wrong prediction     | Write a larger, clearer digit and keep it centered.                       |

---

## 📈 Model Information

| Property        | Value   |
| --------------- | ------- |
| Dataset         | MNIST   |
| Classes         | 10      |
| Image Size      | 28 × 28 |
| Training Images | 0,000  |
| Test Images     | 10,000  |
| CNN Accuracy    | ~99%    |

---

## 📸 Screenshots

<img width="804" height="509" alt="Screenshot 2026-06-26 214228" src="https://github.com/user-attachments/assets/75fde529-21f5-477a-89cf-469fecf7ca13" />





## 🔮 Future Improvements

* Support handwritten letters (EMNIST)
* Multiple digit recognition
* Custom dataset training
* GPU acceleration
* Save prediction history
* Mobile deployment

