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
from matplotlib.animation import FuncAnimation
look_back = 200

def floatconv(pulse):
	for i in pulse:
		if i !='':
			float(i)
		else:
			del i
	return pulse

def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


def rewrite():
	fo  = open('pulse.txt','r')
	n = open('pulse.txt', 'a')
	pulse=fo.read().split(' ')
	rm=open('pulse.txt', 'w').close()
	del pulse[-1],pulse[0:999]
	for i in pulse:
		n.write(i+' ')



def updateData(num):
	#del firstdata
	fo  = open('pulse.txt','r')
	pulse=fo.read().split(' ')
	num=-num-200
	del pulse[-1],pulse[0:num]
	pulse=floatconv(pulse)
	#time=[]
	a=0
	'''for i in pulse:
		time.append(round(a,1))
		a+=0.3'''
	pulse=numpy.array(pulse)
	pulse= numpy.reshape(pulse, (-1, 1))
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(pulse)

	# split into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	# reshape into X=t and Y=t+1
	#look_back = 200#predict for future 1 minute
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	datasetX, datasetY = create_dataset(dataset, look_back)
	oritestX=datasetX
	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	return trainX,trainY,testX,testY,oritestX,scaler

def trainmodel(trainX,trainY,testX,testY,scaler):
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	#scaler = MinMaxScaler(feature_range=(0, 1))
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
	return model


def predictapp(num,oritestX,model,scaler,a):
	futureX=oritestX[-1:]
	futureX = numpy.reshape(futureX, (futureX.shape[0], 1, futureX.shape[1]))
	futurePredict=model.predict(futureX)
	futurePredict=scaler.inverse_transform(futurePredict)
	futurePredict=futurePredict.ravel()
	print 'fu',futurePredict
	return futurePredict
plt.ion()
model=[]
oripredict=[]
oriplot=[]
x1=0
t,v=[],[]
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line, = ax1.plot(t,v, linestyle="-", color="r")
def p(a, b):
    t.append(a)
    v.append(b)
    ax1.set_xlim(min(t), max(t) + 1)
    ax1.set_ylim(0, 70)
    line.set_data(t, v)
    plt.pause(0.5)
    ax1.figure.canvas.draw()
# shift train predictions for plotting
def plotpulse(num):
	global x1
	trainX,trainY,testX,testY,oritestX,scaler=updateData(num)
	if num==-201: 
		model.append(trainmodel(trainX,trainY,testX,testY,scaler))
		#futurePredict=predictapp(num,oritestX,model[0],scaler,0)
		'''futurePredictPlot = numpy.empty_like(futurePredict)
		futurePredictPlot[:, :] = numpy.nan
		futurePredictPlot[num:, :] = futurePredict
		print 'le', len(futurePredictPlot)
		oriplot.append(futurePredictPlot)
		xaxis=numpy.arange(0,0.3*201/60,0.3/60)
		oriplot.append(xaxis)'''
	futurePredict=predictapp(num,oritestX,model[0],scaler,0)
	p(x1,futurePredict)
	x1=x1+0.005
	#plt.show()
	'''fig = plt.figure()
	fig.canvas.set_window_title('Predicted Heart Rate')
	plt.plot(xaxis,futurePredictPlot)
	#fig.suptitle('test title', fontsize=20)
	plt.xlabel('Time', fontsize=12)
	plt.ylabel('Pulse', fontsize=12)
	#fig.savefig('test.jpg')
	#plt.show(block=True)
	plt.pause(0.1)'''
