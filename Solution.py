import tensorflow
import numpy
import pandas
import matplotlib.pyplot
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def scale_column(train_data, test_data, column):
    min_value = train_data[column].min()
    max_value = train_data[column].max()
    train_data[column] = (train_data[column] - min_value)/(max_value - min_value)
    test_data[column] = (test_data[column] - min_value)/(max_value - min_value)


#read csv into a Pandas DataFrame df
df = pandas.read_csv('Iris.csv')
print(df.head())
rows, cols = df.shape
print('Rows:', rows)
print('Columns:', cols)

#transform the Species columns into numbers to introduce them in the neural network
label_names = df['Species'].unique()
index_and_label = list(enumerate(label_names))
print(index_and_label)

label_to_index = dict((label, index) for index, label in index_and_label)
print(label_to_index)

df = df.replace(label_to_index)

#the csv is ordered by species, so we shuffle it
#frac=1 means that all suffle rows will be returned
df = df.sample(frac=1)


#Splitting the suffled dataset for trainig set and test set
# 80% * dataset_rows = 120  ---> train_data
# 20% * dataset)rows = 30   ---> test_data
train_data = df.iloc[:120, :]
test_data = df.iloc[120:, :]

#validation data
x_train = train_data.iloc[:120, 1:-1]
y_train = train_data.iloc[:120, -1:]
x_test = test_data.iloc[120:, 1:-1]
y_test = test_data.iloc[120:, -1:]

# make all values between 0 and 1
scale_column(x_train, x_test, 'SepalLengthCm')
scale_column(x_train, x_test, 'SepalWidthCm')
scale_column(x_train, x_test, 'PetalLengthCm')
scale_column(x_train, x_test, 'PetalWidthCm')

#Convert data to arrays in order to be given as parameters to the Neural Network
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

#Make the Neural Network
model = tensorflow.keras.Sequential()

#Input Layer
#All columns except ID and Species
model.add(tensorflow.keras.layers.Input(shape=[4]))
#Hidden layer with 64 neurons and the activation function 'relu'
model.add(tensorflow.keras.layers.Dense(units=64, activation='relu'))
#Output Layer with 3 neurons(3 Iris Species) and activation function 'softmax' (used for output layers with more than 2 neurons)
model.add(tensorflow.keras.layers.Dense(units=3, activation='softmax'))

#The loss function
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Training
#Interate 75 times
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=75)

# #Evaluate the training by giving two data samples that are unknown
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(test_acc)
#
#
# #Plotting the training and validation loss
# matplotlib.pyplot.figure(figsize=(20, 10))
# matplotlib.pyplot.subplot(2, 2, 1)
# matplotlib.pyplot.plot(history.history['loss'])
# matplotlib.pyplot.subplot(2, 2, 2)
# matplotlib.pyplot.plot(history.history['accuracy'])
# matplotlib.pyplot.subplot(2, 2, 3)
# matplotlib.pyplot.plot(history.history['val_loss'])
# matplotlib.pyplot.subplot(2, 2, 4)
# matplotlib.pyplot.plot(history.history['val_accuracy'])
# matplotlib.pyplot.show()
