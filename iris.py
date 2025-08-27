# ---------------------------
# 1. Load and explore dataset
# ---------------------------

import kaggle
# kaggle.api.authenticate()
# kaggle.api.dataset_download_files("uciml/iris", path = ".", unzip=True)

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Iris.csv")

# exploring data
print(df.head())
df = df.drop(columns=["Id"])

print(df.describe())

print(df.info())
# no null values

print(df['Species'].value_counts())
# each species has 50 samples


# ---------------------------
# 2. Visualizations
# ---------------------------

# histogram of each property
df.hist()
plt.show()

# scatterplot
sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species')
plt.show()

sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species')
plt.show()
#Iris Setosa is separate each time

sns.scatterplot(data=df, x='SepalLengthCm', y='PetalLengthCm', hue='Species')
plt.show()

sns.scatterplot(data=df, x='SepalWidthCm', y='PetalWidthCm', hue='Species')
plt.show()

print(df.corr(numeric_only=True))

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()


# ---------------------------
# 3. Preprocessing
# ---------------------------

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])

print(df['Species'].value_counts())
# converted to 0, 1, 2



# ---------------------------
# 4. Train/test split
# ---------------------------
from sklearn.model_selection import train_test_split
# train
# test
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
# 70-30 is the standard split for small datasets


# ---------------------------
# 5. Models and evaluation
# ---------------------------

# Basic Classification model - Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

print("Logistic Regression Accuracy", model.score(x_test, y_test))


# K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier()

model_knn.fit(x_train, y_train)
print("KNN Accuracy: ", model_knn.score(x_test, y_test))


# SVC
from sklearn.svm import SVC

model_svc = SVC()
model_svc.fit(x_train, y_train)
print("SVM Accuracy: ", model_svc.score(x_test, y_test))


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier()
model_dt.fit(x_train, y_train)
print("Decision Tree Accuracy:", model_dt.score(x_test, y_test))


from sklearn.model_selection import cross_val_score
# Cross validation to give a better estimate of how accurate model will be
scores = cross_val_score(model, X, Y, cv=5) # 5 fold cross validation
print("Cross validation scores:", scores)
print("Average cross validation score:", scores.mean())

from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(x_test)
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
