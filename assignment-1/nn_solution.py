import re
from stop_words import get_stop_words
import csv
from random import randrange
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing import text, sequence
from keras.utils import np_utils

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|[\d-]")


def clean_review(review):
    review = REPLACE_NO_SPACE.sub("", review.lower())
    review = REPLACE_WITH_SPACE.sub(" ", review)

    return review



stop_words = get_stop_words('en')
stop_words = get_stop_words('english')


def get_content(each_content):

    #   each_content = ['I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air conditioned theater and watching a light-hearted comedy. The plot is simplistic, but the dialogue is witty and the characters are likable (even the well bread suspected serial killer). While some may be disappointed when they realize this is not Match Point 2: Risk Addiction, I thought it was proof that Woody Allen is still fully in control of the style many of us have grown to love.<br /><br />This was the most I\'d laughed at one of Woody\'s comedies in years (dare I say a decade?). While I\'ve never been impressed with Scarlet Johanson, in this she managed to tone down her "sexy" image and jumped right into a average, but spirited young woman.<br /><br />This may not be the crown jewel of his career, but it was wittier than "Devil Wears Prada" and more interesting than "Superman" a great comedy to go see with friends.', 'positive']
    probable_review = each_content[0]
    sentiment = each_content[1]
    probable_review = clean_review(probable_review)
    review_words = probable_review.split(' ')

    for review_word in review_words:
        if review_word in stop_words:
            review_words.remove(review_word)

    probable_review = ' '.join(review_words)

    return [probable_review, sentiment]


with open('/home/rongon/workspace/assignment-1/dataset/IMDB Dataset.csv', 'r') as file:
    my_reader = csv.reader(file, delimiter=',')

    line_counter = 0
    word_count_per_line = []
    start = True
    negative_reviews = []
    positive_reviews = []

    for row in my_reader:
        if line_counter == 0 and start:
            start = False
            continue

        data = get_content(row)

        cleaned_review, sentiment = data
        #            cleaned_review = clean_review(review)
        cleaned_review_words = cleaned_review.split(' ')
        cleaned_review_words = ' '.join(cleaned_review_words).split()

        word_count_per_line.append(len(cleaned_review_words))

        if sentiment == 'negative':
            negative_reviews.append(cleaned_review_words)
        elif sentiment == 'positive':
            positive_reviews.append(cleaned_review_words)
        line_counter += 1

    test_data_position = randrange(10)
    counter = 0

    positive_test_data = []
    negative_test_data = []

    positive_train_data = []
    negative_train_data = []

    for positive_review in positive_reviews:
        if test_data_position == counter:
            positive_test_data.append(positive_review)
        else:
            positive_train_data.append(positive_review)

        if counter % 9 == 0:
            counter = 0
            test_data_position = randrange(9)
        else:
            counter = counter + 1

    counter = 0

    for negative_review in negative_reviews:
        if test_data_position == counter:
            negative_test_data.append(negative_review)
        else:
            negative_train_data.append(negative_review)

        if counter % 9 == 0:
            counter = 0
            test_data_position = randrange(9)
        else:
            counter = counter + 1

    del positive_reviews
    del negative_reviews

    positive_test_sentiment = len(positive_test_data) * [1]
    negative_test_sentiment = len(negative_test_data) * [0]

    positive_train_sentiment = len(positive_train_data) * [1]
    negative_train_sentiment = len(negative_train_data) * [0]

    training_x_dl = positive_train_data + negative_train_data
    training_y_dl = positive_train_sentiment + negative_train_sentiment

    del positive_train_data
    del negative_train_data

    del positive_train_sentiment
    del negative_train_sentiment

    testing_x_dl = positive_test_data + negative_test_data
    testing_y_dl = positive_test_sentiment + negative_test_sentiment

    del positive_test_data
    del negative_test_data

    del positive_test_sentiment
    del negative_test_sentiment

    print(len(training_x_dl))

maxlen = np.percentile(word_count_per_line, 95)
maxlen = int(maxlen)

with open('/home/rongon/workspace/assignment-1/dataset/movie_review.csv', 'r') as file:
    my_reader = csv.reader(file, delimiter=',')
    line_counter = 0
    WORD_LIMIT = 30
    words = []

    for row in my_reader:
        if line_counter == 0 and start:
            start = False
            continue
        nltk_review = row[4]
        nltk_review = clean_review(nltk_review)
        nltk_review_words = nltk_review.split(' ')

        for nltk_review_word in nltk_review_words:
            if nltk_review_word in stop_words:
                nltk_review_words.remove(nltk_review_word)

        if len(nltk_review_words) >= WORD_LIMIT:
            words = words + nltk_review_words

    words = list(set(words))
    total_words_length = len(words)

maxIndex = maxlen
starting_index = 0
embedding_data=[]

while total_words_length > maxIndex:
    data = words[starting_index:maxIndex]
    temp_data = []
    for word in data:
        if len(word) !=0:
            temp_data.append(word)
    embedding_data.append(temp_data)
    starting_index = maxIndex
    maxIndex = maxIndex+ maxlen

maxIndex = maxIndex - maxlen
if total_words_length > maxIndex:
    data = words[maxIndex:total_words_length]
    temp_data = []
    for word in data:
        if len(word) != 0:
            temp_data.append(word)

    embedding_data.append(temp_data)

print(embedding_data[0])

min_count=1 # word frequency grater or equal 'min_count' can be embedded
size=300 # word vector/'embedding size'
workers=4
window = 4

model = Word2Vec(embedding_data, size=size, window=window, min_count=1, workers=workers)
word_vector = model.wv


max_features = 100000
embed_size = 300

"""
RUN 1: padding with average number of words per line
RUN 2: padding with 95th percentile number of words words count list
"""
tok = text.Tokenizer(num_words=max_features,lower=True)
tok.fit_on_texts(training_x_dl + testing_x_dl)
X_train = tok.texts_to_sequences(training_x_dl)
X_test = tok.texts_to_sequences(testing_x_dl)

x_train = sequence.pad_sequences(X_train,maxlen=maxlen, padding='post', truncating='post', dtype='int32')
x_test = sequence.pad_sequences(X_test,maxlen=maxlen,padding='post', truncating='post', dtype='int32')

training_y = np_utils.to_categorical(training_y_dl)
testing_y = np_utils.to_categorical(testing_y_dl)


word_index = tok.word_index
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
total_excluded = []
for word, i in word_index.items():
    if i >= max_features:
        continue
    try:
        embedding_vector = word_vector.get_vector(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except:
        total_excluded.append([word,i])
        pass

from keras.layers import Embedding
from keras.layers import Bidirectional,GRU,Input, Dense
from keras.models import Model
from keras.optimizers import Adam
model = None
sequence_input = Input(shape=(maxlen, ))
x = Embedding(num_words, embed_size, weights=[embedding_matrix], trainable = True)(sequence_input)
x = Bidirectional(GRU(128, return_sequences=False,dropout=0.1,recurrent_dropout=0.1))(x)
preds = Dense(2, activation="softmax")(x)
model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])


batch_size = 256
epochs = 5
from keras.utils import np_utils
training_y = np_utils.to_categorical(training_y_dl)
testing_y = np_utils.to_categorical(testing_y_dl)
val_index = int(len(x_train) * 0.8)
x_val = x_train[val_index:]
y_val = training_y[val_index:]

x_train = x_train[:val_index]
training_y = training_y[:val_index]


model.fit(x_train, training_y, batch_size=batch_size, validation_data=(x_val, y_val), epochs=epochs,verbose=1)

y_pred = model.predict(x_test,batch_size=1024,verbose=1)
estimated_y = np.argmax(y_pred, axis = 1)


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

precision = precision_score(testing_y_dl, estimated_y, average='weighted')
print('Precision: %.3f' % precision)

recall = recall_score(testing_y_dl, estimated_y, average='weighted')
print('Recall: %.3f' % recall)

score = f1_score(testing_y_dl, estimated_y, average='weighted')
print('F-Measure: %.3f' % score)

#maxlen = average words per line
Precision: 0.718
Recall: 0.701
F_Measure: 0.695

#maxlen=95 percentile
Precision: 0.715
Recall: 0.701
F_Measure: 0.696

test_sample_1 = "This movie is was fantastic. I will go there again"
test_sample_2 = " Highly_recommended."
test_sample_3 = " May be I like this movie. Can't assure "
test_sample_4 = " I will think twice before recommending"
test_sample_5 = " Not to my taste. Will skip and watch another movie."
test_sample_6 = " The movie really sucks!"

test_samples = [test_sample_1,test_sample_2, test_sample_3,test_sample_4,test_sample_5,test_sample_6]

test_samples_token = tok.texts_to_sequences(test_samples)
test_samples_token_pad = sequence.pad_sequences(test_samples_token,maxlen=maxlen, dtype='int32')
output = model.predict(x = test_samples_token_pad)
output = np.argmax(output, axis = 1)

# output = array([1, 1, 1, 0, 0, 0])