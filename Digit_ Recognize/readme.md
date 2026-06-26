
# 🎨 Handwritten Digit Recognizer

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A simple desktop application for recognizing handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** trained on the **MNIST** dataset.

Users can draw digits with the mouse, predict the digit using a trained model, visualize the processed image, and save their drawings for future use.

---

## ✨ Features

* 🖌️ Draw digits with the mouse
* 🧠 CNN-based digit recognition
* 📊 Prediction confidence score
* 🖼️ Preview of the processed 28×28 image
* 💾 Save drawings with predicted labels
* 🧹 Clear the drawing canvas
* ⚡ Lightweight and easy to use



## ⚙️ Requirements

* Python 3.9+
* TensorFlow
* OpenCV
* NumPy
* Matplotlib

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

If you don't already have the trained model, run:

```bash
python train_mnist.py
```

This will:

* Download the MNIST dataset
* Train the CNN model
* Save the model as:

```text
mnist_cnn.h5
```

---

##  Run the Application

```bash
python digit_recognizer.py
```

---

## 🎮 Controls

| Key   | Action                                    |
| ----- | ----------------------------------------- |
| **P** | Predict the drawn digit                   |
| **S** | Save the drawing with its predicted label |
| **C** | Clear the drawing canvas                  |
| **Q** | Quit the application                      |

---

## 🖱️ How to Use

1. Launch the application.
2. Draw a digit (0–9) using your mouse.
3. Press **P** to predict the digit.
4. View the prediction and confidence score.
5. Press **S** to save the drawing.
6. Press **C** to clear the canvas and draw another digit.
7. Press **Q** to exit.

---

## ⚙️ How It Works

1. Create a blank 280×280 drawing canvas.
2. Draw a digit using the mouse.
3. Resize the drawing to 28×28 pixels.
4. Normalize pixel values.
5. Feed the processed image into the CNN model.
6. Display:

   * Predicted digit
   * Confidence score
   * Processed 28×28 image
7. Optionally save the drawing with the predicted label.

---

## 📊 Example Output

```text
Predicted: 8
Confidence: 99.72%

Saved as drawings/digit_3_8.png
```

---


## 📈 Model Information

| Property        | Value    |
| --------------- | -------- |
| Dataset         | MNIST    |
| Classes         | 10 (0–9) |
| Input Size      | 28 × 28  |
| Training Images | 60,000   |
| Test Images     | 10,000   |
| CNN Accuracy    | ~99%     |

---

## 💡 Tips

* Draw the digit in the center of the canvas.
* Use thick, continuous strokes.
* Avoid drawing multiple digits at once.
* Clear the canvas before drawing a new digit.

---

##  Troubleshooting

| Problem               | Solution                                                     |
| --------------------- | ------------------------------------------------------------ |
| Failed to load model  | Ensure `mnist_cnn.h5` is in the project directory.           |
| Incorrect predictions | Draw a larger, clearer digit in the center of the canvas.    |
| Nothing saved         | Predict the digit first by pressing **P**, then press **S**. |

---

## 🔮 Future Improvements

* Better image preprocessing
* Undo/redo functionality
* Adjustable brush size
* Support for letters (EMNIST)
* Export prediction history
* Modern graphical user interface (GUI)


