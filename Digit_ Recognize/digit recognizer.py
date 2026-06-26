

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model

# ---------- Settings ----------
CANVAS_SIZE = 280
BRUSH_THICKNESS = 20
MODEL_PATH = "mnist_cnn.h5"
SAVE_DIR = "drawings"

# ---------- Load Model ----------
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print("❌ Failed to load model:", e)
    print(f"   Make sure '{MODEL_PATH}' is in the same folder as this script.")
    raise SystemExit(1)

# ---------- Drawing State ----------
drawing = False
ix, iy = -1, -1
img = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

last_label = None  # remembers the most recent prediction, used by 's' to save


def draw_digit(event, x, y, flags, param):
    """Mouse callback: draws a smooth line on the canvas while the
    left mouse button is held down."""
    global ix, iy, drawing, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(img, (ix, iy), (x, y), 255, BRUSH_THICKNESS, cv2.LINE_AA)
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), 255, BRUSH_THICKNESS, cv2.LINE_AA)


def preprocess_image(image):
    """Resize to 28x28, normalize, and shape for the model.
    MNIST digits are white-on-black, so invert if the canvas
    average suggests the opposite."""
    resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    if np.mean(resized) > 127:
        resized = 255 - resized
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, 28, 28, 1)


def predict_digit(image):
    """Returns (predicted_label, confidence) for the given canvas image."""
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0)[0]
    predicted_label = int(np.argmax(prediction))
    confidence = float(prediction[predicted_label])
    return predicted_label, confidence


def show_prediction(image, label, confidence):
    """Pops up a small window showing what the model actually saw,
    alongside its prediction and confidence."""
    plt.figure(figsize=(2, 2))
    plt.imshow(preprocess_image(image).reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {label} ({confidence * 100:.2f}%)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_image(image, label):
    """Saves the raw 280x280 canvas to disk, labeled with the prediction."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    count = len(os.listdir(SAVE_DIR))
    filename = os.path.join(SAVE_DIR, f"digit_{count}_{label}.png")
    cv2.imwrite(filename, image)
    print(f"✅ Saved as {filename}")


def main():
    global img, last_label

    window_name = "Draw a digit (0-9)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_digit)

    print("🖌️  Instructions:")
    print("  - Draw a digit using the mouse.")
    print("  - Press 'p' to predict the digit.")
    print("  - Press 's' to save the last drawing + prediction.")
    print("  - Press 'c' to clear the canvas.")
    print("  - Press 'q' to quit.")

    while True:
        # Overlay key instructions on a copy so they don't get drawn onto the real canvas
        display_img = img.copy()
        cv2.putText(
            display_img,
            "'p'=predict 's'=save 'c'=clear 'q'=quit",
            (10, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            255,
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            img = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
            last_label = None

        elif key == ord("p"):
            label, confidence = predict_digit(img)
            last_label = label
            print(f"🔢 Predicted: {label} with {confidence * 100:.2f}% confidence")
            show_prediction(img, label, confidence)

        elif key == ord("s"):
            if last_label is not None:
                save_image(img, last_label)
            else:
                print("⚠️  Nothing to save yet — press 'p' to predict first.")

        elif key == ord("q"):
            print("👋 Exiting...")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()