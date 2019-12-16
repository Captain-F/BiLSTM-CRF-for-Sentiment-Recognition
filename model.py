from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Bidirectional, LSTM, GRU, RNN, Flatten
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
import numpy as np
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.layers import TimeDistributed
from keras_self_attention import SeqSelfAttention

model = Sequential()
model.add(Bidirectional(LSTM(128, input_shape = (50, 100),
                             return_sequences = True)))
model.add(Dropout(0.5))
model.add(Dense(3, activation = 'softmax'))

crf = CRF(3, sparse_target=True)
model.add(crf)
model.compile(optimizer = 'adam', loss= crf_loss, metrics= [crf_viterbi_accuracy])
history = model.fit(train_X_, train_Y_, epochs = 150, validation_data = (test_X_, test_Y_), 
                    batch_size = 128, verbose = 1)

y_pre = model.predict(test_X_, batch_size= 128)
y_pre = [i[0].argmax(axis = 0) for i in y_pre]
print(classification_report(test_Y.argmax(axis= 1), y_pre, digits= 5))
