# Iris Dataset Analysis

This project explores and models the classic **Iris dataset**, one of the most well-known datasets in machine learning.  
The dataset contains **150 flower samples** across **3 species** (*Setosa, Versicolor, Virginica*) with **4 measured features**: sepal length, sepal width, petal length, and petal width.  

The goal is to perform **exploratory data analysis (EDA)** and evaluate multiple **classification models**.

---

## Dataset
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) / [Kaggle](https://www.kaggle.com/datasets/uciml/iris)  
- 150 samples, evenly distributed across 3 species.  
- 4 numerical features + 1 categorical label (species).  

---

## Project Workflow
1. **Exploratory Data Analysis**
   - Summary statistics
   - Histograms & scatterplots
   - Correlation heatmap  
2. **Data Preprocessing**
   - Dropped irrelevant columns (`Id`)
   - Label encoding of categorical target (`Species`)  
3. **Modeling**
   - Logistic Regression  
   - K-Nearest Neighbors (KNN)  
   - Support Vector Machine (SVM)  
   - Decision Tree  
4. **Evaluation**
   - Train/test split (70/30)  
   - Accuracy scores  
   - 5-fold cross-validation  
   - Confusion matrix & classification report  

---

## Results
| Model                | Accuracy (approx) |
|-----------------------|-------------------|
| Logistic Regression   | ~95% |
| KNN                  | ~95% |
| SVM                  | ~95% |
| Decision Tree         | ~95% |

- **All models perform well**
- Confusion matrix shows that *Setosa* is perfectly separable, while *Versicolor* and *Virginica* occasionally overlap.  

---

## Tech Stack
- Python 3.12  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---

## How to Run
Clone the repo and install dependencies:
```bash
git clone https://github.com/your-username/iris-analysis.git
cd iris-analysis
pip install -r requirements.txt
python analysis.py
