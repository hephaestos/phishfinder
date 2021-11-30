from json import dump, load
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import os

from nltk.util import pad_sequence

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
        if len(tokens) > 0:
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
    return min([len(email) for email in data])

def calc_vocab_size(tokenizer: Tokenizer):
    return len(tokenizer.word_index) + 1 # Add 1 since index begins at 0

def encode_pad_text(tokenizer: Tokenizer, data, max_length):
    encoded_data = tokenizer.texts_to_sequences(data)
    padded_data = pad_sequences(encoded_data, max_length, padding='post')
    return padded_data

train_real = create_tokens_list(DIRECTORY_REAL_TRAIN)
train_phish = create_tokens_list(DIRECTORY_PHISH_TRAIN)
train_data = train_real + train_phish
save_data([train_data, train_labels], 'train.pkl')

test_real = create_tokens_list(DIRECTORY_REAL_TEST)
test_phish = create_tokens_list(DIRECTORY_PHISH_TEST)
test_data = test_real + test_phish
save_data([test_data, test_labels], 'test.pkl')


# tokens_train, labels_train = load_data('train.pkl')
# tokens_test, labels_test = load_data('test.pkl')




