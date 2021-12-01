from json import dump, load
from keras.layers.normalization.batch_normalization import BatchNormalization
import nltk
from keras.preprocessing.text import Tokenizer
import keras.losses as losses
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.merge import concatenate
import tensorflow as tf
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import os
from numpy import array

# nltk.download('all') # Uncomment if running script for first time - install dependencies
# REFERENCE: https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/
# This reference was used to outline how data was cleaned and prepared for input into the model

DIRECTORY_REAL_TRAIN = r'../data/train_real'
DIRECTORY_REAL_TEST = r'../data/test_real'
DIRECTORY_PHISH_TRAIN = r'../data/train_phish'
DIRECTORY_PHISH_TEST = r'../data/test_phish'
HTML_TAGS = ["a","abbr","acronym","area","b","base","bdo","big","blockquote","body","br","button","caption","cite","code","col","colgroup","dd","del","dfn","div","dl","DOCTYPE","dt","em","fieldset","form","h1","h2","h3","h4","h5","h6","html","hr","i","img","input","ins","kbd","label","legend","li","link","meta","noscript","ol","optgroup","p","param","pre","q","samp","script","select","span","style","sub","sup","table","tbody","td","textarea","tfoot","th","thead","title","tr","tt","ul","var"]

train_labels = [0 for _ in range(3160)] + [1 for _ in range(3160)] # Split data 80/20 between train and testing
test_labels =  [0 for _ in range(790)] + [1 for _ in range(790)]

# Iterate through all of the real email files and tokenize the text, find unique words
def create_tokens_list(directory):
    tokens_list = list()
    for file in os.scandir(directory):
        curr_file = (open(file, mode='r', encoding='ascii', errors="surrogateescape").read())
        tokens = nltk.word_tokenize(curr_file)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if not (word in stop_words or word in HTML_TAGS)]
        tokens = [word for word in tokens if (len(word) > 1 and len(word) < 15) and 'enron' not in word.lower()]
        tokens_list.append(tokens)
    return tokens_list

def save_data(data, filename):
    dump(data, open(filename, 'w'))

def load_data(filename):
    return load(open(filename, 'r'))

def phish_tokenizer(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    return tokenizer

def max_words(data):
    return max([len(email) for email in data])

def calc_vocab_size(tokenizer: Tokenizer):
    return len(tokenizer.word_index) + 1 # Add 1 since index begins at 0

def encode_pad_text(tokenizer: Tokenizer, data, max_length):
    encoded_data = tokenizer.texts_to_sequences(data)
    padded_data = pad_sequences(encoded_data, max_length, padding='post')
    return padded_data


def create_model(max_length, vocab_size):
     # Model w/ some reference to Nvidia Deep Learning parameters
     #inputs = Input(shape=(max_length,))
     #embedding = Embedding(vocab_size, 100)(inputs)
     #conv1 = Conv1D(75, 4, activation='relu')(embedding)
     #drop1 = Dropout(0.4)(conv1)
     #pool1 = MaxPooling1D(pool_size=2)(drop1)
     #flat1 = Flatten()(pool1)
     #conv2 = Conv1D(50, 4, activation='relu')(pool1)
     #drop2 = Dropout(0.4)(conv2)
     #pool2 = MaxPooling1D(pool_size=2)(drop2)
     #flat2 = Flatten()(pool2)
     #conv3 = Conv1D(25, 4, strides=1, activation='relu', padding='same')(flat2)
     #drop3 = Dropout(0.2)(conv3)
     #pool3 = MaxPooling1D(pool_size=2, padding='same')(drop3)
     #flat3 = Flatten()(pool3)
     #dense1 = Dense(10, activation='relu')(flat2)
     #output = Dense(1, activation='sigmoid')(dense1)
     #model = Model(inputs=inputs, outputs=output)
     #model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
     #print(model.summary())
     #return model

     # Model taken from://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentime    nt-analysis/ewBoard->mat[i] = newBoard->grid + (i * newBoard->ncols);
     inputs1 = Input(shape=(max_length,)) # Input size equal to file with greatest word length

     # This layer learns using combinations of 4 words
     embedding1 = Embedding(vocab_size, 100)(inputs1)
     conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
     drop1 = Dropout(0.5)(conv1)
     pool1 = MaxPooling1D(pool_size=2)(drop1)
     flat1 = Flatten()(pool1)

     # This layer uses 6 word combinations
     inputs2 = Input(shape=(max_length,))
     embedding2 = Embedding(vocab_size, 100)(inputs2)
     conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
     drop2 = Dropout(0.5)(conv2)
     pool2 = MaxPooling1D(pool_size=2)(drop2)
     flat2 = Flatten()(pool2)

     # This layer uses 8 word combinations
     inputs3 = Input(shape=(max_length,))
     embedding3 = Embedding(vocab_size, 100)(inputs3)
     conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
     drop3 = Dropout(0.5)(conv3)
     pool3 = MaxPooling1D(pool_size=2)(drop3)
     flat3 = Flatten()(pool3)

     merged = concatenate([flat1, flat2, flat3]) # Combine the results of all 3 layers

     dense1 = Dense(10, activation='relu')(merged)
     outputs = Dense(1, activation='sigmoid')(dense1) # Output as binary classification in dense layer
     model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

     model.compile(loss='binary_crossentropy',
                 optimizer='adam', metrics=['accuracy'])

     print(model.summary()) # Model composition
     return model
   
    
train_real = create_tokens_list(DIRECTORY_REAL_TRAIN)
train_phish = create_tokens_list(DIRECTORY_PHISH_TRAIN)
train_data = train_real + train_phish
save_data([train_data, train_labels], 'train.pkl')

test_real = create_tokens_list(DIRECTORY_REAL_TEST)
test_phish = create_tokens_list(DIRECTORY_PHISH_TEST)
test_data = test_real + test_phish
save_data([test_data, test_labels], 'test.pkl')


tokens_train, labels_train = load_data('train.pkl')
tokens_test, labels_test = load_data('test.pkl')
labels_train = tf.one_hot(labels_train, depth=2)
labels_test = tf.one_hot(labels_test, depth=2)

tokenizer = phish_tokenizer(tokens_train)
max_length = max_words(tokens_train)
vocab_size = calc_vocab_size(tokenizer)
train_X = encode_pad_text(tokenizer, tokens_train, max_length)
print(train_X.shape)
model = create_model(max_length, vocab_size)
model.fit([train_X, train_X, train_X], array(train_labels), epochs=10, batch_size=16) # For training sequence CNN model
# model.fit([train_x], array(train_labels), epochs=20, batch_size=16) # For training w/out sequence CNN model
model.save('phish_model_sequence') # Save sequence model
# model.save('phish_model') # Save non-sequence model
