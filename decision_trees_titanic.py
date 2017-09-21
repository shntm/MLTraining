import pandas as pd
import numpy as np
from sklearn import tree, preprocessing, metrics, linear_model
import graphviz

#defining samples to test with
samplesToTest = 200
samplesToTest = -1*samplesToTest

#load dataset from CSV
dataset = pd.read_csv('train-1.csv')

#load feature names to list
feature_names = list(dataset)

#filling empty spots. Must impute or remove later
dataset["Embarked"] = dataset["Embarked"].fillna('0')
dataset["Age"] = dataset["Age"].fillna(-1)

#converting dataset to numpy array
datasetMatrix = dataset.as_matrix()
datasetArr = np.array(datasetMatrix)

#defining target data i.e. survival of passenger
targetData = np.asarray(datasetArr[:,1], dtype=np.float64)

#label encoding for string values
targetClassName = ['died', 'survived']
embarkPointArray = ['S', 'Q', 'C', '0']
genderArray = ['male', 'female']
usableData = np.delete(datasetArr,[0,1,3,8,10], 1)
featureNames = np.delete(feature_names,[0,1,3,8,10])
lblEncoder = preprocessing.LabelEncoder()
lblEncoderForGender = preprocessing.LabelEncoder()
lblEncoder.fit(embarkPointArray)
lblEncoderForGender.fit(genderArray)
usableData[:,-1] = lblEncoder.transform(usableData[:,-1]);
usableData[:,1] = lblEncoderForGender.transform(usableData[:,1])

#all sample values to float
usableData = np.asarray(usableData, dtype=np.float64)

#dividing to training and test data
trainingData = usableData[:samplesToTest,:]
testData = usableData[samplesToTest:,:]
targetTrainingData = targetData[:samplesToTest]
targetTestData = targetData[samplesToTest:]

#Decision Tree classifier prediction
clftree = tree.DecisionTreeClassifier()
clftree.fit(trainingData, targetTrainingData)
predictionDataTree = clftree.predict(testData)

#Logistic regression classifier prediction
clrLoRegr = linear_model.LogisticRegression()
clrLoRegr.fit(trainingData, targetTrainingData)
predictionDateLoRegr = clrLoRegr.predict(testData)

#predictions and target data
print(predictionDataTree)
print(predictionDateLoRegr)
print(targetTestData)

#metrics for comparision between both classifiers
print(metrics.accuracy_score(targetTestData,predictionDataTree))
print(metrics.confusion_matrix(targetTestData, predictionDataTree))
print(metrics.accuracy_score(targetTestData,predictionDateLoRegr))
print(metrics.confusion_matrix(targetTestData, predictionDateLoRegr))

#export decision tree to pdf
# dot_data = tree.export_graphviz(clf, out_file=None, feature_names = featureNames)
# graph = graphviz.Source(dot_data)
# graph.render("titanic_tree")