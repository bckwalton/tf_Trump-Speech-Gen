import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils

text=(open("./ST.txt").read())
text=text.lower()

#Get Characters in text. Generate a number for each letter and vice versa
characters = sorted(list(set(text)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}
X = []
Y = []
length = len(text)
seq_length = 100
for i in range(0, length-seq_length, 1):
    sequence = text[i:i + seq_length]
    label =text[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])

#Reshaping X and Y groupings
X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)

#Loading model
model = Sequential()
model.add(LSTM(700, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#Train model
model.fit(X_modified, Y_modified, epochs=1, batch_size=100)

#Save model weights
model.save_weights('./text_generator_700_0.2_700_0.2_baseline.h5')

#Load model weights
model.load_weights('./text_generator_700_0.2_700_0.2_baseline.h5')

string_mapped = X[99]
full_string = [n_to_char[value] for value in string_mapped]
# generating characters
for i in range(400):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(characters))

    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    full_string.append(n_to_char[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

text_file = open("Output.txt", "w")
text_file.write(txt)
text_file.close()

#combining text#combinin
txt=""
for char in full_string:
    txt = txt+char
print(txt)
