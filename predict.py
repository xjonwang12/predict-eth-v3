from tensorflow import keras

model = keras.models.load_model('model.h5')
iterations = 12
output = [str(model.predict([i])[0][0]) for i in range(iterations)]
open('output.csv', 'x')
with open('output.csv', 'w') as file:
	file.write(', '.join(output))
