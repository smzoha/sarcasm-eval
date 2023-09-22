import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('Reading data from file')
train_data = pd.read_csv('../clean_data/train_task_a.csv')
test_data = pd.read_csv('../clean_data/test_task_a.csv')

print("=================================================")

print('Sarcastic Tweet Count:')
print(train_data.groupby('sarcastic')['sarcastic'].count())

print("=================================================")

print('Splitting train, validate and test data')

X = np.array(train_data['tweet']).astype('str')
y = np.array(train_data['sarcastic']).astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=False)

print('Split dataset to 60-40 splits (70: training + validation; 30 - test)')

print("=================================================")

print('Creating vectorization layer and adapt over training data')

vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=10000, output_mode='int')
vectorization_layer.adapt(X_train)

print('Size of vocabulary', len(vectorization_layer.get_vocabulary()))

print("=================================================")

print('Creating model')

model = tf.keras.Sequential([
    vectorization_layer,
    tf.keras.layers.Embedding(input_dim=len(vectorization_layer.get_vocabulary()), output_dim=64),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, dropout=0.5, kernel_regularizer=tf.keras.regularizers.l2(0.5),
                             recurrent_regularizer=tf.keras.regularizers.l2(0.5))),
    tf.keras.layers.Dense(1, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.F1Score(threshold=0.5)])

print(model.summary())

print("=================================================")

print('Fitting model to train data')

history = model.fit(X_train, np.array(y_train), epochs=10, validation_split=0.3)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validate'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Chart')
plt.show()

print('Evaluating model with test data from training dataset')

loss, f1score = model.evaluate(X_test, np.array(y_test))
print(loss, f1score)

print("=================================================")

print('Evaluating model with test data')

test_x = np.array(test_data['tweet']).astype('str')
test_y = np.array(test_data['sarcastic']).astype('float32')

loss, f1score = model.evaluate(test_x, test_y)
print(loss, f1score)
