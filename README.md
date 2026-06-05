# CodeAlpha_Handwritten_Character_Recognition
```markdown
# Handwritten Character Recognition (Neural Network)

## 📌 Project Overview
This repository hosts a computer vision and deep-pattern classification pipeline engineered to interpret and identify handwritten numerical characters. Utilizing a classic neural network architecture, the system maps pixel intensity grids into precise, single-digit classifications.

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.14
* **Frameworks & Frameworks:** Scikit-Learn, NumPy, OpenML API

## 📊 Pipeline Architecture & Training Logic
1. **Automated Data Retrieval:** Integrates the OpenML API to dynamically fetch clean subsets of the foundational **MNIST Handwritten Digits** dataset.
2. **Image Preprocessing:** Normalizes spatial 2D matrix arrays by scaling raw 8-bit grayscale pixel intensities ($[0, 255]$) down to bounded continuous value intervals ($[0, 1]$), stabilizing neural network gradient calculations.
3. **Network Architecture:** Implements a feedforward **Multi-Layer Perceptron (MLP) Neural Network** structured with dual hidden layers:
   * **Hidden Layer 1:** 100 Neurons capturing localized structural configurations (edges, curves).
   * **Hidden Layer 2:** 50 Neurons abstracting high-level geometric combinations.
4. **Optimization Logic:** Leverages the **Adam optimizer** with stochastic gradient updates, integrated cross-entropy loss tracking, and regularized batch processing over 20 iterations.

## 🚀 Execution Instructions
Ensure you are inside the root repository directory, then run:
```bash
python3 -m pip install -r requirements.txt --break-system-packages
python3 src/train.py
