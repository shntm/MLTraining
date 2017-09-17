from sklearn import datasets, metrics
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy
data = datasets.load_breast_cancer()
radiusData = data.data[:,0]
areaData = data.data[:,3]
targetData = data.target
radiusInput = numpy.reshape(radiusData[:-50], (-1,1))
areaInput = numpy.reshape(areaData[:-50], (-1,1))
combinedInput = numpy.concatenate((radiusInput, areaInput),axis=1)
testRadiusData =  numpy.reshape(radiusData[-50:], (-1,1))
testAreaData =  numpy.reshape(areaData[-50:], (-1,1))
combinedTest = numpy.concatenate((testRadiusData, testAreaData),axis=1)
print(combinedTest)
targetInput = targetData[:-50]
testTargetData = targetData[-50:]
model = linear_model.LogisticRegression()
model.fit(combinedInput, targetInput)
predOutput = model.predict(combinedTest)
print(predOutput)
print(testTargetData)
print(metrics.classification_report(testTargetData, predOutput))
print(metrics.confusion_matrix(testTargetData, predOutput))
# plt.scatter(radiusData, targetData)
# plt.show()
# print(data.data[:,0])