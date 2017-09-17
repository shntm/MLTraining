from sklearn import datasets, metrics
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy
data = datasets.load_breast_cancer()
radiusData = data.data[:,0]
targetData = data.target
radiusInput = numpy.reshape(radiusData[:-50], (-1,1))
testRadiusData =  numpy.reshape(radiusData[-50:], (-1,1))
targetInput = targetData[:-50]
testTargetData = targetData[-50:]
model = linear_model.LogisticRegression()
model.fit(radiusInput, targetInput)
predOutput = model.predict(testRadiusData)
print(predOutput)
print(testTargetData)
print(metrics.classification_report(testTargetData, predOutput))
print(metrics.confusion_matrix(testTargetData, predOutput))
# plt.scatter(radiusData, targetData)
# plt.show()
# print(data.data[:,0])