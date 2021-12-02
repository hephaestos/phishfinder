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
from keras.models import load_model
import tensorflow as tf
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import os
import numpy as np
from numpy import array
# nltk.download('all') # Uncomment if running script for first time - install dependencies

# Authors: Brandon Thomas & Daniel Floyd
# Date: 12/2/2021
# Class: CIS 365 - Artificial Intelligence (Final Project)
# REFERENCE: https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/
# This reference was used to outline how data was cleaned and prepared for input into the model

# Constant directory values for reading in data through script
DIRECTORY_REAL_TRAIN = r'../data/train_real'
DIRECTORY_REAL_TEST = r'../data/test_real'
DIRECTORY_PHISH_TRAIN = r'../data/train_phish'
DIRECTORY_PHISH_TEST = r'../data/test_phish'
DIRECTORY_DEMO_PHISH = r'../data/demo_data/phishing3'
DIRECTORY_DEMO_REAL = r'../data/demo_data/prof_emails'

HTML_TAGS = ["a", "abbr", "acronym", "area", "b", "base", "bdo", "big", "blockquote", "body", "br", "button", "caption", "cite", "code", "col", "colgroup", "dd", "del", "dfn", "div", "dl", "DOCTYPE", "dt", "em", "fieldset", "form", "h1", "h2", "h3", "h4", "h5", "h6", "html", "hr",
             "i", "img", "input", "ins", "kbd", "label", "legend", "li", "link", "meta", "noscript", "ol", "optgroup", "p", "param", "pre", "q", "samp", "script", "select", "span", "style", "sub", "sup", "table", "tbody", "td", "textarea", "tfoot", "th", "thead", "title", "tr", "tt", "ul", "var"]

# Split data 80/20 between train and testing
train_labels = [0 for _ in range(3160)] + [1 for _ in range(3160)]
test_labels = [0 for _ in range(790)] + [1 for _ in range(790)]

demo_labels = [0 for _ in range(6)] + [1 for _ in range(6)]


# Iterate through all of the real email files and tokenize the text, find unique words
def create_tokens_list(directory):
    tokens_list = list()
    for file in os.scandir(directory):
        curr_file = (open(file, mode='r', encoding='ascii', 
                     errors="surrogateescape").read()) # Open next file in directory
        tokens = nltk.word_tokenize(curr_file) # Break down into simple words
        tokens = [word.lower() for word in tokens if word.isalpha()] # Make sure it actually can be a word
        stop_words = set(stopwords.words('english')) # Remove extraneous stopwords using corpus
        tokens = [word for word in tokens if not (
            word in stop_words or word in HTML_TAGS)] # Remove pointless HTML tags
        tokens = [word for word in tokens if (
            len(word) > 1 and len(word) < 15) and 'enron' not in word.lower()] # Remove Enron specific data and small data
        tokens_list.append(tokens)
    return tokens_list # Return the dataset (list of lists)


def save_data(data, filename):
    dump(data, open(filename, 'w'))


def load_data(filename):
    return load(open(filename, 'r'))


# Creates Keras tokenizer to assign integer values to words
def phish_tokenizer(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    return tokenizer


# Function determines max words in an email over a given dataset
def max_words(data):
    return max([len(email) for email in data])


# Calculate the max amount of unique words for the given tokenizer (dataset)
def calc_vocab_size(tokenizer: Tokenizer):
    return len(tokenizer.word_index) + 1  # Add 1 since index begins at 0


# Function to pad all of the inputs (Train, test, demo, etc.) to CNN so they are all the same length
def encode_pad_text(tokenizer: Tokenizer, data, max_length):
    encoded_data = tokenizer.texts_to_sequences(data) # Convert to integer values based on word dictionary for tokenizer 
    padded_data = pad_sequences(encoded_data, max_length, padding='post')
    return padded_data


def create_model(max_length, vocab_size):
    '''
     # Model w/ some reference to Nvidia Deep Learning parameters
     inputs = Input(shape=(max_length,))
     embedding = Embedding(vocab_size, 100)(inputs)
     conv1 = Conv1D(75, 4, activation='relu')(embedding)
     drop1 = Dropout(0.4)(conv1)
     pool1 = MaxPooling1D(pool_size=2)(drop1)
     flat1 = Flatten()(pool1)
     conv2 = Conv1D(50, 4, activation='relu')(pool1)
     drop2 = Dropout(0.4)(conv2)
     pool2 = MaxPooling1D(pool_size=2)(drop2)
     flat2 = Flatten()(pool2)
     conv3 = Conv1D(25, 4, strides=1, activation='relu', padding='same')(flat2)
     drop3 = Dropout(0.2)(conv3)
     pool3 = MaxPooling1D(pool_size=2, padding='same')(drop3)
     flat3 = Flatten()(pool3)
     dense1 = Dense(10, activation='relu')(flat2)
     output = Dense(1, activation='sigmoid')(dense1)
     model = Model(inputs=inputs, outputs=output)
     model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
     print(model.summary())
     return model
     '''

    # Model taken from://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/
    # Input size equal to file with greatest word length
    inputs1 = Input(shape=(max_length,))

    # This layer learns using combinations of 4 words
    embedding1 = Embedding(vocab_size, 100)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1) # Drop out half of the nodes to avoid overfitting
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

    # Combine the results of all 3 layers
    merged = concatenate([flat1, flat2, flat3])

    dense1 = Dense(10, activation='relu')(merged)
    # Output as binary classification in dense layer
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print(model.summary())  # Model composition
    return model


train_real = create_tokens_list(DIRECTORY_REAL_TRAIN)
train_phish = create_tokens_list(DIRECTORY_PHISH_TRAIN)
train_data = train_real + train_phish
save_data([train_data, train_labels], 'train.pkl')

test_real = create_tokens_list(DIRECTORY_REAL_TEST)
test_phish = create_tokens_list(DIRECTORY_PHISH_TEST)
test_data = test_real + test_phish
save_data([test_data, test_labels], 'test.pkl')

demo_real = create_tokens_list(DIRECTORY_DEMO_REAL)
demo_phish = create_tokens_list(DIRECTORY_DEMO_PHISH)
demo_data = demo_real + demo_phish[:6] # Demo data is professors emails first (6) followed by 6 phishing emails
save_data([demo_data, demo_labels], 'demo.pkl')


# These lines not necessary, but good to show how to save and load this data if necessary
tokens_train, labels_train = load_data('train.pkl')  
tokens_test, labels_test = load_data('test.pkl')
tokens_demo, labels_demo = load_data('demo.pkl')
# labels_train = tf.one_hot(labels_train, depth=2) # Use if model complains about shape for input data
# abels_test = tf.one_hot(labels_test, depth=2)


tokenizer = phish_tokenizer(tokens_train) # Tokenize based on training data
max_length = max_words(tokens_train) + 6 # Offset by 6 - artifact of building on linux & running on different pc
vocab_size = calc_vocab_size(tokenizer) # Max amount of unique words in all of training set
train_X = encode_pad_text(tokenizer, tokens_train, max_length) # Padded train data
test_X = encode_pad_text(tokenizer, tokens_test, max_length) # Padded test data
demo_X = encode_pad_text(tokenizer, tokens_demo, max_length) # Padded demo data ready for fitting

### CODE TO CREATE & FIT MODELS - Uncomment if need to build model files ###
# model = create_model(max_length, vocab_size)
# model.fit([train_X, train_X, train_X], array(train_labels), epochs=10, batch_size=16) # For training sequence CNN model
# model.fit([train_x], array(train_labels), epochs=20, batch_size=16) # For training w/out sequence CNN model
# model.save('phish_model_sequence') # Save sequence model
# model.save('phish_model') # Save non-sequence model

# Load saved models
load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost') # Loading on machine separate from where model was built
phish_model = load_model('../models/phish_model/', options=load_options)
channels_model = load_model('../models/phish_model_channels', options=load_options)


# Evaluate channel model on both training and test datasets
loss_channel_train, acc_channel_train = channels_model.evaluate([train_X, train_X, train_X], array(train_labels), verbose=0)
loss_channel_test, acc_channel_test = channels_model.evaluate([test_X, test_X, test_X], array(test_labels), verbose=0)
print('Channel model train accuracy: ' + str(round(acc_channel_train * 100, 2)))
print('Channel mode train loss: ' + str(loss_channel_train))
print('Channel model test accuracy: ' + str(round(acc_channel_test * 100, 2)))
print('Channel model test loss: ' + str(loss_channel_test))

# Evaluate non-channel model on both training and test datasets
loss_non_channel_train, acc_non_channel_train = phish_model.evaluate([train_X], array(train_labels), verbose=0)
loss_non_channel_test, acc_non_channel_test = phish_model.evaluate([test_X], array(test_labels), verbose=0)
print('Non-channel model train accuracy: ' + str(round(acc_non_channel_train * 100, 2)))
print('Non-channel mode train loss: ' + str(loss_non_channel_train))
print('Non-channel model test accuracy: ' + str(round(acc_non_channel_test * 100, 2)))
print('Non-channel model test loss: ' + str(loss_non_channel_test))

### Demo test ###
preds = channels_model.predict([demo_X, demo_X, demo_X]) # Demo data (First 6 prof, last 6 phish)
print('0 indicates the email is legitimate, 1 indicates a phishing email')
for num in preds:
    print("Prediction is: ", num)

