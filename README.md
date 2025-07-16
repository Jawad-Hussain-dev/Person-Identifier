# 🧠 Person Identifier using HOG + Decision Tree

This project implements a lightweight and interpretable **Person Identification System** using **HOG (Histogram of Oriented Gradients)** features and a **Decision Tree Classifier**. Built for the course *Introduction to Computaional Thinking*, the focus is on explainability, interpretability, and modular thinking.

## 🎯 Project Goals

- Apply **Computational Thinking (CT)** principles:
  - Decomposition
  - Pattern Recognition
  - Abstraction
  - Algorithmic Thinking
- Use **HOG** to extract meaningful facial features
- Train a **Decision Tree** classifier for clear, explainable predictions
- Provide a **GUI** for user-friendly testing of predictions
- Visualize decision paths and most important features for interpretability

## 📁 Project Structure

```

Person-Identifier-HOG/
├── src/                      # All Python source files
│   ├── feature\_extraction.py
│   ├── dataset\_builder.py
│   ├── decision\_tree.py
│   ├── train2.py
│   ├── test\_predict.py
│   └── gui\_predict.py
│
├── models/                   # Trained models
│   ├── tree\_model.pkl
│   └── hog\_dataset.pkl
│
├── data/                     # Optional: sample data for demo
│
├── report.pdf                # IEEE format project report 
├── README.md                 # This file

````

## 🧪 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Person-Identifier-HOG.git
cd Person-Identifier-HOG
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install opencv-python scikit-image scikit-learn matplotlib pillow joblib
```

### 3. Train the model

```bash
python src/train2.py
```

This builds the dataset, trains the Decision Tree, and saves it in `/models`.

### 4. Launch the GUI

```bash
python src/gui_predict.py
```

Use the GUI to:

* Upload an image
* Get the predicted person ID
* See the decision path
* View the original + HOG visualizations

## 📊 Sample Output

* ✅ Accuracy: 97.4%
* 🧠 Model: Decision Tree (depth=10)
* 📌 Explanation: Decision path and top features
* 🖼 GUI: HOG + Original image display side-by-side

## 🧠 CT Principles in Action

| Principle            | Application                                     |
| -------------------- | ----------------------------------------------- |
| Decomposition        | Separated modules for extraction, training, GUI |
| Pattern Recognition  | HOG identifies facial structure & orientation   |
| Abstraction          | Faces turned into numeric feature vectors       |
| Algorithmic Thinking | Decision Tree rules used to predict & explain   |

## 📄 Report

For full details, methodology, and results, see:
📎 **[report.pdf](./report.pdf)**

## 👥 Team Members

* Jawad Hussain
* Abdullah Siraj Khan
* Abdul Hadi Javed


```
