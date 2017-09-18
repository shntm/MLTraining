from sklearn import datasets, linear_model
import numpy as numpy
from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plotter

# load the diabetes dataset
diabetes = datasets.load_diabetes()
bmiList = []
bpList = []
targetList = []

#add feature data to respective list
for i in range(len(diabetes.data)):
	bmiList.append(diabetes.data[i][2])
	bpList.append(diabetes.data[i][3])
	targetList.append(diabetes.target[i])

#reshape lists to numpy vector array also keep 5 inputs for samples accuracy
bmiTestInput = numpy.reshape(bmiList[:-5],(-1,1))
bpTestInput = numpy.reshape(bpList[:-5],(-1,1))

#inputMatrix for BP and BMI features combined
inputMatrix = numpy.concatenate((bmiTestInput,bpTestInput), axis=1)

#prediction inputs for BP and BMI features separately
bmiPredInput = numpy.reshape(bmiList[-5:],(-1,1))
bpPredInput = numpy.reshape(bpList[-5:],(-1,1))

#prediction inputs for BP and BMI features combined
predInputMatrix = numpy.concatenate((bmiPredInput,bpPredInput), axis=1)

#actual prediction outputs to test accuracy
testOut = numpy.reshape(targetList[:-5],(-1,1))

#load the model
regr = linear_model.LinearRegression()
regr2 = linear_model.LinearRegression();

#fit  model with just BMI
regr2.fit(bmiTestInput, targetList[:-5])

#fit  model with both features
regr.fit(inputMatrix,targetList[:-5])

#model's prediction for both features combined
finalPredCombined = regr.predict(predInputMatrix)

#model's prediction for only BMI inputs
finalPredBMI = regr2.predict(bmiPredInput)

#mean squared errors for both predictions for comparision
print("Mean squared error for combined: %.2f" % mean_squared_error(targetList[-5:], finalPredCombined))
print("Mean squared error for BMI: %.2f" % mean_squared_error(targetList[-5:], finalPredBMI))
# plotter.scatter(bpList, targetList)
# plotter.xlabel("BMI of patient");
# plotter.show("Target vector");
