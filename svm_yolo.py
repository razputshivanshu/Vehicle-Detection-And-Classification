import cv2
import torch
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load pre-trained YOLO model (YOLOv5 as an example)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to detect objects using YOLO
def detect_objects_yolo(image):
    results = model(image)
    detections = results.xyxy[0]  # xyxy format: [x1, y1, x2, y2, confidence, class]
    return detections.cpu().numpy()

# Function to extract features for SVM classification
def extract_features_from_detections(detections):
    features = []
    labels = []
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        width = x2 - x1
        height = y2 - y1
        area = width * height
        features.append([conf, width, height, area])
        labels.append(int(class_id))  # Using YOLO's class_id as labels
    return features, labels

# Function to preprocess image and extract features using YOLO for SVM
def process_image_for_svm(image):
    detections = detect_objects_yolo(image)
    features, labels = extract_features_from_detections(detections)
    return features, labels

# Example: Load dataset of images and extract YOLO features
def load_images_and_extract_features(image_paths):
    all_features = []
    all_labels = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        features, labels = process_image_for_svm(image)
        all_features.extend(features)
        all_labels.extend(labels)
    return np.array(all_features), np.array(all_labels)

# Train SVM classifier on YOLO-extracted features
def train_svm_classifier(X_train, y_train):
    clf = svm.SVC(kernel='linear')  # Using linear kernel for SVM
    clf.fit(X_train, y_train)
    return clf

# Evaluate SVM classifier
def evaluate_svm_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == "__main__":
    # Example image paths (replace with your image dataset)
    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']

    # Load images, detect objects using YOLO, and extract features
    X, y = load_images_and_extract_features(image_paths)

    # Split the data into training and testing sets for SVM
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the SVM classifier
    svm_classifier = train_svm_classifier(X_train, y_train)

    # Evaluate the classifier
    evaluate_svm_classifier(svm_classifier, X_test, y_test)
