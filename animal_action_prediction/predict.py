import os
import pandas as pd

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

print(train['Action'].value_counts())
print(train['Action'].unique())