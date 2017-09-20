import pandas as pd
import numpy as np
from sklearn import tree, preprocessing
import graphviz
dataset = pd.read_csv('train-1.csv')
feature_names = list(dataset)
dataset["Embarked"] = dataset["Embarked"].fillna('0')
dataset["Age"] = dataset["Age"].fillna(-1)
dataset.describe();
datasetMatrix = dataset.as_matrix()
datasetArr = np.array(datasetMatrix)
targetData = np.asarray(datasetArr[:,1], dtype=np.float64)
targetClassName = ['died', 'survived']
embarkPointArray = ['S', 'Q', 'C', '0']
genderArray = ['male', 'female']
usableData = np.delete(datasetArr,[0,1,3,8,10], 1)
featureNames = np.delete(feature_names,[0,1,3,8,10])
lblEncoder = preprocessing.LabelEncoder()
lblEncoderForGender = preprocessing.LabelEncoder()
lblEncoder.fit(embarkPointArray)
lblEncoderForGender.fit(genderArray)
# print(usableData[:,2])
usableData[:,-1] = lblEncoder.transform(usableData[:,-1]);
usableData[:,1] = lblEncoderForGender.transform(usableData[:,1])
# print(usableData[:,1])
# print(lblEncoderForGender.inverse_transform(usableData[:,1].astype(int)))
usableData = np.asarray(usableData, dtype=np.float64)
# print(usableData)
trainingData = usableData[:-5,:]
testData = usableData[-5:,:]
targetTrainingData = targetData[:-5]
targetTestData = targetData[-5:]

print(featureNames)
clf = tree.DecisionTreeClassifier()
clf.fit(trainingData, targetTrainingData)
predictionData = clf.predict(testData)
print(predictionData)
print(targetTestData)
dot_data = tree.export_graphviz(clf, out_file=None, feature_names = featureNames)
graph = graphviz.Source(dot_data)
graph.render("titanic_tree")