# src/train.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import fetch_openml

def load_and_preprocess_mnist():
    print("Fetching a clean subset of the MNIST handwritten digits dataset from OpenML...")
    # Fetching a smaller subset (10,000 samples) to ensure it downloads and trains lightning-fast on your MacBook
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    
    # Take a subset for rapid training execution
    X_subset = X[:10000]
    y_subset = y[:10000]
    
    # Normalize pixel values from [0, 255] to [0, 1]
    X_normalized = X_subset / 255.0
    
    return X_normalized, y_subset

if __name__ == "__main__":
    # 1. Load data
    X, y = load_and_preprocess_mnist()
    
    # 2. Split into Train/Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Initialize and Train the Multi-Layer Perceptron (Neural Network)
    print("Training Handwritten Digit Neural Network (MLPClassifier)...")
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50), 
        max_iter=20, 
        alpha=1e-4,
        solver='adam', 
        verbose=True, 
        random_state=42, 
        learning_rate_init=.001
    )
    model.fit(X_train, y_train)
    
    # 4. Predict & Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n================ EVALUATION METRICS ================")
    print(classification_report(y_test, predictions))
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    print("====================================================")