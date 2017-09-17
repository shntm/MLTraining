from sklearn import datasets, linear_model
import numpy as numpy
from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plotter

diabetes = datasets.load_diabetes()
bmiList = []
bpList = []
targetList = []
for i in range(len(diabetes.data)):
	bmiList.append(diabetes.data[i][2])
	bpList.append(diabetes.data[i][3])
	targetList.append(diabetes.target[i])
bmiTestInput = numpy.reshape(bmiList[:-5],(-1,1))
bpTestInput = numpy.reshape(bpList[:-5],(-1,1))
inputMatrix = numpy.concatenate((bmiTestInput,bpTestInput), axis=1)
bmiPredInput = numpy.reshape(bmiList[-5:],(-1,1))
bpPredInput = numpy.reshape(bpList[-5:],(-1,1))
predInputMatrix = numpy.concatenate((bmiPredInput,bpPredInput), axis=1)
testOut = numpy.reshape(targetList[:-5],(-1,1))
regr = linear_model.LinearRegression()
regr2 = linear_model.LinearRegression();
regr2.fit(bmiTestInput, targetList[:-5])
regr.fit(inputMatrix,targetList[:-5])
finalPredCombined = regr.predict(predInputMatrix)
finalPredBMI = regr2.predict(bmiPredInput)
print("Mean squared error for combined: %.2f" % mean_squared_error(targetList[-5:], finalPredCombined))
print("Mean squared error for BMI: %.2f" % mean_squared_error(targetList[-5:], finalPredBMI))
# plotter.scatter(bpList, targetList)
# plotter.xlabel("BMI of patient");
# plotter.show("Target vector");
