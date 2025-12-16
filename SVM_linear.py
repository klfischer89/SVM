import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("./data/diabetes.csv")

X = df[["BMI", "Age"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)

sc = StandardScaler()
sc.fit(X_train)

model = LinearSVC(max_iter = 1000000, C = 10)
model.fit(sc.transform(X_train), y_train)

print(model.score(sc.transform(X_test), y_test))

bmi_min = df["BMI"].min()
bmi_max = df["BMI"].max()

age_min = df["Age"].min()
age_max = df["Age"].max()

bmi_range = np.arange(bmi_min, bmi_max, 0.1)
age_range = np.arange(age_min, age_max, 0.1)

xx, yy = np.meshgrid(bmi_range, age_range)

X_pred = np.c_[xx.ravel(), yy.ravel()]
zz = model.predict(sc.transform(X_pred)).reshape(xx.shape)

plt.contourf(xx, yy, zz, alpha = 0.5)
plt.scatter(X_train["BMI"], X_train["Age"], c = y_train, s = 5)