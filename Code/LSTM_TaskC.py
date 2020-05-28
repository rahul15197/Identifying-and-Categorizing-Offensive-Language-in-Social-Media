import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Input, Embedding, CuDNNLSTM
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

file = open('classifier_dump_c', 'rb')
file1 = open('accuracy_list_dump_c', 'rb')
classifier_list = pickle.load(file)
accuracy_list = pickle.load(file1)

olid_data = open("training.tsv", encoding='utf-8')
taskc_test = open("taskc_test.tsv", encoding='utf-8')
taskc_test_labels = open("labels_c.csv", encoding='utf-8')

training_dataframe = pd.read_table(olid_data, sep="\t")
test_dataframe = pd.read_table(taskc_test, sep="\t")
test_labels_dataframe = pd.read_csv(taskc_test_labels)

training_dataframe = training_dataframe.dropna(subset=["subtask_c"])

training_dataframe['target'] = training_dataframe.subtask_c.astype('category').cat.codes
test_labels_dataframe['target'] = test_labels_dataframe.subtask_c.astype('category').cat.codes

num_class = len(np.unique(training_dataframe.subtask_c.values))
Y_train = training_dataframe['target'].values
Y_test = test_labels_dataframe['target'].values

print(training_dataframe)

MAX_LENGTH = 50
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_dataframe.tweet.values)
train_tweet_seq = tokenizer.texts_to_sequences(training_dataframe.tweet.values)
train_tweet_seq_padded = pad_sequences(train_tweet_seq, maxlen=MAX_LENGTH)
test_tweet_seq = tokenizer.texts_to_sequences(test_dataframe.tweet.values)
test_tweet_seq_padded = pad_sequences(test_tweet_seq, maxlen=MAX_LENGTH)

vocab_size = len(tokenizer.word_index) + 1

inputs = Input(shape=(MAX_LENGTH,))
embedding_layer = Embedding(vocab_size, 20, input_length=MAX_LENGTH)(inputs)
x = CuDNNLSTM(32)(embedding_layer)
x = Dense(64, activation='relu')(x)
x = Dropout(0.9)(x)
predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
epochs = 5

model_history = model.fit([train_tweet_seq_padded], batch_size=128, y=to_categorical(Y_train), verbose=1, epochs=epochs)
predicted = model.predict(test_tweet_seq_padded)
predicted = np.argmax(predicted, axis=1)
print("Accuracy for Training set (LSTM)")
for e in range(epochs):
    print("Epoch " + str(e) + ": Accuracy = " + str(round(model_history.history['acc'][e] * 100, 2)))
accuracy_lstm = round(accuracy_score(Y_test, predicted) * 100, 2)
print("\nClassification report for LSTM")
print(classification_report(Y_test, predicted))
print("Accuracy for Test set (LSTM)= " + str(accuracy_lstm) + " %")
print("\nConfusion Matrix (LSTM)")
cf_matrix_lstm = confusion_matrix(Y_test, predicted)
print(cf_matrix_lstm)

labels = ['GRP', 'IND', 'OTH']

# graph plot lstm
fig = plt.figure(1)
ax = fig.add_subplot()
cax = ax.matshow(cf_matrix_lstm)
plt.title('Confusion matrix of SubTask C (LSTM)')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')

classifier_list.append("LSTM")
accuracy_list.append(accuracy_lstm)

fig = plt.figure(2)
ax = fig.add_subplot()
plt.title("Accuracy Comparison for SubTask C")
plt.ylabel('Classifier')
plt.xlabel('Accuracy (%)')
plt.barh(classifier_list, accuracy_list)
for i, j in enumerate(accuracy_list):
    ax.text(j + 1, i + .10, str(j), color='blue')
plt.show()
