import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("./train.csv")

test = pd.read_csv("./test.csv")

train.info()

train.head()

train.drop(['Id'], axis=1, inplace=True)

train.describe()

train.isnull().sum()

train.duplicated().sum()

train.quality.value_counts()

train.quality.value_counts().plot(kind='bar')

train.corr()

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
for feature in features:
    plt.figure(figsize=(8, 6))
    plt.bar(train[feature], train['quality'])
    plt.xlabel(feature)
    plt.ylabel('Quality')
    plt.title(f'Comparison of {feature} to Quality')
    plt.show()
    print(train.quality.corr(train[feature]))

train.drop(['density'], axis=1, inplace=True)

train = train.drop(41)
train = train.drop(696)
train = train.drop(1338)
train = train.drop(634)
train = train.drop(1017)
train.describe()

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']

for feature in features:
    plt.figure(figsize=(8, 6))
    plt.bar(train[feature], train['quality'])
    plt.xlabel(feature)
    plt.ylabel('Quality')
    plt.title(f'Comparison of {feature} to Quality')
    plt.show()
    print(train.quality.corr(train[feature]))
