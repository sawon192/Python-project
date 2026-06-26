"""
Handwritten Digit Recognizer (Webcam)
---------------------------------------
Hold up a handwritten digit (0-9) to your webcam inside the green box 
and have a trained MNIST CNN model predict it in real time.

Requirements:
    pip install numpy opencv-python tensorflow

You also need a trained model file named 'mnist_cnn.h5' in the same
folder as this script (not included in this repo).

Controls:
    q - quit
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model

MODEL_PATH = "mnist_cnn.h5"

# Load the trained model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(" Failed to load model:", e)
    print(f"   Make sure '{MODEL_PATH}' is in the same folder as this script.")
    raise SystemExit(1)

def preprocess_image(img):
    """Preprocess image for MNIST prediction"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get binary image
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assuming this is the digit)
        c = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(c)
        
        # Extract ROI
        digit = thresh[y:y+h, x:x+w]
        
        # Resize to 28x28
        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize and reshape for model
        digit = digit.astype('float32') / 255.0
        digit = digit.reshape(1, 28, 28, 1)
        
        return digit, (x, y, w, h)
    
    return None, None

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Could not open webcam. Check that it's connected and not in use.")
    raise SystemExit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame from webcam.")
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Create a rectangle where user should write the digit
    cv2.rectangle(frame, (200, 100), (400, 300), (0, 255, 0), 2)
    roi = frame[100:300, 200:400]
    
    # Preprocess and predict
    processed_digit, bbox = preprocess_image(roi)
    
    if processed_digit is not None:
        # Predict
        prediction = model.predict(processed_digit, verbose=0)
        predicted_label = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        # Draw prediction on frame
        x, y, w, h = bbox
        cv2.putText(frame, f"Prediction: {predicted_label}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Digit Recognition - Write in the green box', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()