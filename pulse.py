# load and plot dataset
import numpy
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pylab
fo  = open('pulse.txt','r')
pulse=fo.read().split(' ')
del pulse[-1]

pulse = [ float(elem) for elem in pulse ]

time=[]
a=0
for i in pulse:
	if a<=1.5:
		print a
	time.append(round(a,1))
	a+=0.3
pulse=numpy.array(pulse)
pulse= numpy.reshape(pulse, (-1, 1))
print pulse[0:5],len(pulse),type(pulse)
#print type(pulse)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(pulse)
print dataset[0:5]
xaxis=numpy.arange(0,0.3*len(pulse),0.3*len(pulse)/len(pulse))
pylab.plot(xaxis,pulse)
#plot1=plt.plot(dataset)
#pylab.ylim([0,1000])
plt.show()
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#print(len(train), len(test))


def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)



# reshape into X=t and Y=t+1
look_back = 200
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print trainX[:5],trainY[:5]
# reshape input to be [samples, time steps, features]
print trainX.shape[0],trainX.shape[1]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#print trainX[:5],trainX.shape[0],trainX.shape[1],trainX.shape[2]


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print 'true'
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict



# plot baseline and predictions
pylab.plot(xaxis,scaler.inverse_transform(dataset))
pylab.plot(xaxis,trainPredictPlot)
pylab.plot(xaxis,testPredictPlot)

plt.show()


