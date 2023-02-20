import time
import numpy
import yfinance
from tensorflow import keras


def fetch_data():
	eth = yfinance.Ticker('ETH-USD')
	data = eth.history(period='1d', interval='1m')
	return data['Close'][-1]


model = keras.models.Sequential([
	keras.layers.Dense(units=1, input_shape=[1])
])

current_price = fetch_data()
xs = numpy.array([i for i in range(20)], dtype=float)
ys = numpy.array([2.7 * i + current_price for i in range(20)], dtype=float)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
model.fit(x=xs, y=ys, epochs=5, verbose=1)
model.summary()

print(model.layers[0].weights)

model.layers[0].weights[0].assign(numpy.array([[2.7]]))
model.layers[0].weights[1].assign(numpy.array([current_price]))

print(model.layers[0].weights)


model.save('model.h5'.format(time.time()))

