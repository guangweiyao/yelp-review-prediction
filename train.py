# get the yelp data from https://github.com/vc1492a/Yelp-Challenge-Dataset/blob/master/Prepped%20Data/output.csv?raw=true

import wandb
from wandb.wandb_keras import WandbKerasCallback

from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

import pandas as pd

run = wandb.init()
config = run.config
summary = run.summary

config.max_words = 1000
config.max_length = 500

df = pd.read_csv('yelp.csv')

text = df['text']
target = df['sentiment']

# Remove the blank rows from the series:
target = target[pd.notnull(text)]
text = text[pd.notnull(text)]

print('Number of rows in the excel sheet is: ', len(text))

category_to_num = {"Negative": 0, "Positive": 1}
target_num = [category_to_num[t] for t in target]
target_one_hot = np_utils.to_categorical(target_num)

tokenizer = Tokenizer(num_words=config.max_words)
tokenizer.fit_on_texts(list(text))
sequences = tokenizer.texts_to_sequences(list(text))
data = pad_sequences(sequences, maxlen=config.max_length)


train_data = data[:200000]
test_data = data[200000:]
train_target = target_one_hot[:200000]
test_target = target_one_hot[200000:]

model = Sequential()
model.add(Embedding(config.max_words, 10, input_length=config.max_length))
model.add(LSTM(10, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(2, activation='sigmoid'))

model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, train_target, batch_size=1000, epochs=5, callbacks=[WandbKerasCallback()], validation_data=(test_data, test_target))



