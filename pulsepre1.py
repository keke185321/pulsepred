# load and plot dataset
import numpy
#import matplotlib.pyplot as plt
import matplotlib.axes as axes
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pylab
from matplotlib.pyplot import show
from matplotlib import pyplot as plt
fo  = open('pulse.txt','r')
pulse=fo.read().split(' ')
del pulse[-1]
def floatconv(pulse):
	for i in pulse:
		if i !='':
			float(i)
		else:
			del i
	return pulse
pulse=floatconv(pulse)
time=[]
a=0
for i in pulse:
	#if a<=1.5:
	#	print a
	time.append(round(a,1))
	a+=0.3
pulse=numpy.array(pulse)
pulse= numpy.reshape(pulse, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(pulse)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


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

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
def plotpulse():
	trainPredictPlot = numpy.empty_like(dataset)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	xaxis=numpy.arange(0,0.3*len(trainPredictPlot)/60,0.3/60)
	fig = plt.figure()
	fig.canvas.set_window_title('Predicted Heart Rate')
	plt.plot(xaxis,trainPredictPlot)
	#fig.suptitle('test title', fontsize=20)
	plt.xlabel('Time', fontsize=12)
	plt.ylabel('Pulse', fontsize=12)
	#fig.savefig('test.jpg')
	plt.show(block=True)
