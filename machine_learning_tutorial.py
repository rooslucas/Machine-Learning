# Python Machine Learning Tutorial
# https://www.youtube.com/watch?v=7eh4d6sabA0

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split  # Used to split data into two sets
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib

# Part one introduction and some fun facts
# df = pd.read_csv('vgsales.csv')
# print(df.shape)  # (16598, 11)
# print(df.describe())  # Describes the mean, count, std etc of the dataset
# print(df.values)  # Outer and inner array

# Part two Real Liufe Problem ~ Music Store
music_data = pd.read_csv('music.csv')
# Division into input and output set
X = music_data.drop(columns=['genre'])
Y = music_data['genre']
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)  # Returns a tuple

model = DecisionTreeClassifier()
model.fit(X, Y)  # Fitting the model to the training set
# predicitions = model.predict(x_test)
# print(predicitions)

# score = accuracy_score(y_test, predicitions)
# print(score)

# joblib.dump(model, 'music-recommender.joblib')  # Can be used to store your model

# Part three Visualize a Decision Tree
tree.export_graphviz(model, 'music-recommender.dot', feature_names=['age', 'gender'], class_names=sorted(Y.unique()),
                     label='all', rounded=True, filled=True)
