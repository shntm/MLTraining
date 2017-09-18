from sklearn import datasets, metrics
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy

#load breast cancer dataset
data = datasets.load_breast_cancer()

#load features into respective arrays
radiusData = data.data[:,0]
areaData = data.data[:,3]
targetData = data.target

#reshape features for sklearn consumption also segregate 50 samples to test accuracy
radiusInput = numpy.reshape(radiusData[:-50], (-1,1))
areaInput = numpy.reshape(areaData[:-50], (-1,1))
combinedInput = numpy.concatenate((radiusInput, areaInput),axis=1)
testRadiusData =  numpy.reshape(radiusData[-50:], (-1,1))
testAreaData =  numpy.reshape(areaData[-50:], (-1,1))
combinedTest = numpy.concatenate((testRadiusData, testAreaData),axis=1)
targetInput = targetData[:-50]
testTargetData = targetData[-50:]

#load model
model = linear_model.LogisticRegression()
model.fit(combinedInput, targetInput)
predOutput = model.predict(combinedTest)

#final prediction vs actual outputs for comparision
print(predOutput)
print(testTargetData)

#reports to measure accuracy
print(metrics.classification_report(testTargetData, predOutput))
print(metrics.confusion_matrix(testTargetData, predOutput))
# plt.scatter(radiusData, targetData)
# plt.show()
# print(data.data[:,0])