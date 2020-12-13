# -*- coding: utf-8 -*-
"""COMP572FinalPart2.ipynb
Original file is located at
    https://colab.research.google.com/drive/1U2760PA-sc9RdJToanFereDtpuozA5xV
"""

"""# 1. (20 points) In this problem, we will learn how to construct a RNN and tune its hyperparameters for a text classification task.

## (a) From keras.datasets import reuters news dataset.
> (Each data point is a sequence of words that are translated into numbers). Set num-words=10000 and testsplit=0.2.
"""

from keras.datasets import reuters
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)

"""## (b) Find number of classes in this problem."""

num_classes = max(y_train) + 1
print(f'Number of classes: {num_classes}')

"""## (c) Use a Recurrent Neural Network (RNN) to solve the problem. 
> (Hint: in your designed network, you can use dropout layer to reduce the over-fitting.
> Please refer to https://keras.io/api/layers/recurrent_layers/ to learn more about the Recurrent layers in keras)
"""

EPOCHS = 10
BATCH_SIZE = 64
ACTIVATION = 'softmax'
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']
NUM_CLASSES = num_classes if num_classes is not None else 46

from keras.preprocessing import sequence
from keras.utils import np_utils as npu
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, Dropout

x_train, x_test = sequence.pad_sequences(x_train, maxlen=500), sequence.pad_sequences(x_test, maxlen=500)
y_train = npu.to_categorical(y_train, num_classes)
y_test = npu.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Embedding(10000, BATCH_SIZE))
model.add(SimpleRNN(BATCH_SIZE or 64, return_sequences=True))
model.add(SimpleRNN(BATCH_SIZE or 64, return_sequences=True))
model.add(SimpleRNN(BATCH_SIZE or 64, return_sequences=True))
model.add(SimpleRNN(BATCH_SIZE or 64))
model.add(Dropout(0.25))
model.add(Dense(46, activation=ACTIVATION or 'softmax'))

model.compile(optimizer=OPTIMIZER or 'rmsprop', loss=LOSS or 'categorical_crossentropy', metrics=METRICS or ['accuracy'])
model.fit(x_train, y_train, epochs=EPOCHS or 10, batch_size=BATCH_SIZE or 64, validation_split=0.3)
model.summary()

"""## (d) Evaluate the performance of your model using the test set."""

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

y_prediction = model.predict(x_test)

mse = mean_squared_error(y_test, y_prediction)
rmse = (np.sqrt(mean_squared_error(y_test, y_prediction)))
r2 = r2_score(y_test, y_prediction)
mae = mean_absolute_error(y_test, y_prediction)

print('Performance score report:\n MSE:  %(mse).4f\n RMSE: %(rmse).4f\n R2:   %(r2).4f\n MAE:  %(mae).4f' % {
    'mse':mse,
    'rmse':rmse,
    'r2':r2,
    'mae':mae
})

"""## (e) Try to solve the same problem with (FFNN) and evaluate the performance of your model using the test set."""

from keras.preprocessing import sequence
from keras.utils import np_utils as npu
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, Dropout, Activation

ffnn_model = Sequential(
    [
        Dense(512, input_shape=(500,)),
        Activation('relu'),
        Dropout(0.5),
        Dense(46),
        Activation('softmax')
    ]
)
ffnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ffnn_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.3)

ffnn_model.summary()

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

y_prediction = ffnn_model.predict(x_test)

mse = mean_squared_error(y_test, y_prediction)
rmse = (np.sqrt(mean_squared_error(y_test, y_prediction)))
r2 = r2_score(y_test, y_prediction)
mae = mean_absolute_error(y_test, y_prediction)

print('Performance score report:\n MSE:  %(mse).4f\n RMSE: %(rmse).4f\n R2:   %(r2).4f\n MAE:  %(mae).4f' % {
    'mse':mse,
    'rmse':rmse,
    'r2':r2,
    'mae':mae
})

"""## (f) Compare your two trained models in **(c)** and **(e)**.

By comparing the performance scores of two models, it is found that Recurrent Neural Network 
outperformed Feed Forward Neural Network on these data.
"""