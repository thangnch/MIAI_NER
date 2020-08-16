import pandas as pd
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional
from keras.models import Model, Input, load_model
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import f1_score
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from keras.preprocessing.text import text_to_word_sequence
import pickle
import os

from cls_sentence import sentence

# Config
batch_size = 64
epochs = 50
max_len = 75
embedding = 40


def load_data(filename='ner_dataset.csv'):
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
    df = df.fillna(method = 'ffill')
    return df


def process_data(df, sentences):
    # Xây dựng vocab cho word và tag
    words = list(df['Word'].unique())
    tags = list(df['Tag'].unique())

    # Tạo dict word to index, thêm 2 từ đặc biệt là Unknow và Padding
    word2idx = {w : i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0

    # Tạo dict tag to index, thêm 1 tag đặc biệt và Padding
    tag2idx = {t : i + 1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0

    # Tạo 2 dict index to word và index to tag
    idx2word = {i: w for w, i in word2idx.items()}
    idx2tag = {i: w for w, i in tag2idx.items()}

    # Chuyển các câu về dạng vector of index
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    # Padding các câu về max_len
    X = pad_sequences(maxlen = max_len, sequences = X, padding = "post", value = word2idx["PAD"])
    # Chuyển các tag về dạng index
    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    # Tiền hành padding về max_len
    y = pad_sequences(maxlen = max_len, sequences = y, padding = "post", value = tag2idx["PAD"])

    # Chuyển y về dạng one-hot
    num_tag = df['Tag'].nunique()
    y = [to_categorical(i, num_classes = num_tag + 1) for i in y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

    # Save data
    return X_train, X_test, y_train, y_test, word2idx, tag2idx, idx2word, idx2tag, num_tag, words, tags


def build_model(num_tags, hidden_size = 50):
    # Model architecture
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=len(words) + 2, output_dim=embedding, input_length=max_len, mask_zero=False)(input)
    model = Bidirectional(LSTM(units=hidden_size, return_sequences=True, recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(hidden_size, activation="relu"))(model)
    crf = CRF(num_tags + 1)  # CRF layer
    out = crf(model)  # output

    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

    model.summary()
    return model


if not os.path.exists("data.pkl"):
    print("Data not found, make it!")
    df = load_data()
    getter = sentence(df)
    sentences = getter.sentences
    X_train, X_test, y_train, y_test, word2idx, tag2idx, idx2word, idx2tag, num_tag,words, tags  = process_data(df, sentences)
    file = open('data.pkl', 'wb')
    data = [X_train, X_test, y_train, y_test, word2idx, tag2idx, idx2word, idx2tag, num_tag, words, tags]
    pickle.dump(data, file)
    file.close()
else:
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
        X_train, X_test, y_train, y_test, word2idx, tag2idx, idx2word, idx2tag, num_tag, words, tags = data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10]


if not os.path.exists("model.hdf5"):
    model = build_model(num_tag)
    checkpoint = ModelCheckpoint(filepath = 'model.hdf5',
                           verbose = 0,
                           mode = 'auto',
                           save_best_only = True,
                           monitor='val_loss')
    history = model.fit(X_train, np.array(y_train), batch_size=batch_size, epochs=epochs,
                        validation_split=0.1, callbacks=[checkpoint])
else:
    model = build_model(num_tag)
    model.load_weights("model.hdf5")

# Test với toàn bộ tập test
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)
y_test_true = np.argmax(y_test, -1)

# Kiểm thử F1-Score
y_pred = [[idx2tag[i] for i in row] for row in y_pred]
y_test_true = [[idx2tag[i] for i in row] for row in y_test_true]
print("F1-score is : {:.1%}".format(f1_score(y_test_true, y_pred)))

# Test với một câu ngẫu nhiên trong tập test
idx = np.random.randint(0,X_test.shape[0])

p = model.predict(np.array([X_test[idx]]))
p = np.argmax(p, axis=-1)
true = np.argmax(y_test[i], -1)

print("Example #{}".format(idx))

print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(40 * "*")
for w, t, pred in zip(X_test[idx], true, p[0]):
    if w != 0:
        print("{:15}: {:5} {}".format(words[w-2], idx2tag[t], idx2tag[pred]))