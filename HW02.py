
'''
HW2
'''


##########Part 0 ###########

'''
    1)  from sklearn.datasets import load_digits  (Each datapoint is a 8x8 image of a digit)
    Split your data into train(80% of data) and test(20% of data) via random selection
     
    2)  Try MLPClassifier from sklearn.neural_network
        (a NN with two hidden layers, each with 100 nodes)
        Use 10% of your training data as the validation set to tune other hyper-parameters. Try different values and pick the best one.
        
    3)  print classification report for the test set
'''
# YOUR CODE GOES HERE
#1
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


digits = load_digits()

X, X_t, y, y_t = train_test_split(digits.data, digits.target, test_size=0.2,random_state = 0)

#2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1,random_state = 0)

mlp_clf = MLPClassifier(hidden_layer_sizes=(100,100,), max_iter=300, activation = 'relu', solver='adam',random_state=0) # test on validation set to tune the hyperparams
mlp_clf.fit(X_train, y_train)

#3
y_pred = mlp_clf.predict(X_t)
print(classification_report(y_t, y_pred))

##########Part 1- NN for Classification ###########

'''
    1)  Try to have the same NN (the same architecture) in Keras. Try different activation functions for hidden layers to get a reasonable network.
    
    Hint: 
    1- use validation set (e.g. 10% of your training data) to pick the best value for the hyper-parameters
    2- you need to convert your labels to vectors of 0s and 1  (Try OneHotEncoder from sklearn.preprocessing)
    
    activation fcn for output layer: sigmoid
   
'''

# YOUR CODE GOES HERE
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# your hyper-params can be different!
model = Sequential()
model.add(Dense(100, input_dim=X.shape[1], activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='sigmoid')) # predictions has a value in [0,1] 


yy = to_categorical(y)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, yy, epochs=10, batch_size=50, validation_split=0.1)

pred = model.predict_classes(X_t)  


from sklearn.metrics import classification_report
print(classification_report(y_t, pred))

'''
    2)  Use 'softmax' activation function in output layer, print the predictions/ what is the difference?
'''
# YOUR CODE GOES HERE
model2 = Sequential()
# your hyper-params can be different!
model2.add(Dense(100, input_dim=X.shape[1], activation='relu'))
model2.add(Dense(100, activation='relu'))
model2.add(Dense(10, activation='softmax')) # predictions are probabilities  (add up to 1)


yy = to_categorical(y)

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.fit(X, yy, epochs=10, batch_size=50, validation_split=0.1)

pred2 = model2.predict_classes(X_t) 

from sklearn.metrics import classification_report
print(classification_report(y_t, pred2))


'''
    3)  save your model as a .h5 (or .hdf5) file
    
'''

# YOUR CODE GOES HERE
model.save('my_model.h5')
model2.save('my_model2.h5')


'''
    4)  load your saved your model and test it using the test set
    
'''
# YOUR CODE GOES HERE
from keras.models import load_model
model11 = load_model('my_model.h5')
model22 = load_model('my_model2.h5')

pred22 = model22.predict_classes(X_t) 
pred11 = model11.predict_classes(X_t)  

print('report for model1:',classification_report(y_t, pred11),'report for model2:',classification_report(y_t, pred22))

########## Part 2- NN for Regression ###########



'''
    1)  from sklearn.datasets import load_boston
    Extract the description of all the features and print it. What is this dataset and what is the ML task here?
    Split your data into train(80% of data) and test(20% of data) via random selection
        
'''

# YOUR CODE GOES HERE
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, Y = load_boston(return_X_y=True)
x, x_test, y, y_test = train_test_split(X, Y, test_size=0.20, random_state=123)

'''
    2)  Try LinearRegression from sklearn.linear_model   
        Try it with and without normalization. Compare the results and pick the best trained model(for comparison try different metrics from sklearn.metrics like: r2, mse, mae)
        (Hint: for normalizing your data set normalize=True)
    
'''

# YOUR CODE GOES HERE

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# Cross Validation Method:Hold-out- CV

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.20, random_state=123)
reg1 = LinearRegression( normalize=False).fit(x_train, y_train)
reg2 = LinearRegression( normalize=True).fit(x_train, y_train)
pred1 = reg1.predict(x_valid)
pred2 = reg2.predict(x_valid)

print('r2 score for reg1:',r2_score(y_valid, pred1),'r2 score for reg2:',r2_score(y_valid, pred2))
print('mean abs err for reg 1:',mean_absolute_error(y_valid, pred1),'mean abs err for reg2:',mean_absolute_error(y_valid, pred2))
print('mean square error for reg 1:', mean_squared_error(y_valid, pred1), 'mean square error for reg 2:', mean_squared_error(y_valid, pred2))
# they are almost the same


'''
    3)  print the performance of your trained model on the test set
'''
# YOUR CODE GOES HERE

pred = reg1.predict(x_test)
print('r2 score for knn:','r2 score for reg:',r2_score(y_test, pred))
print('mean abs err for reg:',mean_absolute_error(y_test, pred))
print('mean square error for reg:', mean_squared_error(y_test, pred))



'''
    4)  Repeat Q2 and Q3 with the Normalized data
'''
# YOUR CODE GOES HERE


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
x = scaler.fit_transform(x)
x_test = scaler.transform(x_test)
# repeat the whole code with the new X values

'''
    5)  Use a simple NN (with 2-3 FC layers) in keras. Try to tune the hyper-parametrs using your validation set. Print the performance of your trained model on the test set.
'''
# YOUR CODE GOES HERE

X, Y = load_boston(return_X_y=True)
x, x_test, y, y_test = train_test_split(X, Y, test_size=0.20, random_state=123)

# your hyper-params can be different!

model = Sequential()
model.add(Dense(units=256, activation='relu', input_shape=(x.shape[1],)))
model.add(Dense(units=32, activation='linear'))
model.add(Dense(units=1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x, y, epochs=50, validation_split=0.1, batch_size=15)


pred = model.predict(x_test)
print('r2 score for knn:','r2 score for reg:',r2_score(y_test, pred))
print('mean abs err for reg:',mean_absolute_error(y_test, pred))
print('mean square error for reg:', mean_squared_error(y_test, pred))



