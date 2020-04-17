import re
from stop_words import get_stop_words
import csv
from random import randrange
from sklearn.linear_model import LogisticRegression

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
    total_words = 0
    start = True
    negative_reviews = []
    positive_reviews = []

    for row in my_reader:
        if line_counter == 0 and start:
            start = False
            continue

        data = get_content(row)

        cleaned_review, sentiment = data

        if sentiment == 'negative':
            negative_reviews.append(cleaned_review)
        elif sentiment == 'positive':
            positive_reviews.append(cleaned_review)
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

    training_x = positive_train_data + negative_train_data
    training_y = positive_train_sentiment + negative_train_sentiment

    del positive_train_data
    del negative_train_data

    del positive_train_sentiment
    del negative_train_sentiment

    testing_x = positive_test_data + negative_test_data
    testing_y = positive_test_sentiment + negative_test_sentiment

    del positive_test_data
    del negative_test_data

    del positive_test_sentiment
    del negative_test_sentiment

    print(len(training_x))


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(min_df=5, ngram_range=(1, 1))
X_train = vect.fit(training_x).transform(training_x)
X_test = vect.transform(testing_x)

print("Vocabulary size: {}".format(len(vect.vocabulary_)))
print("X_train:\n{}".format(repr(X_train)))
print("X_test: \n{}".format(repr(X_test)))

feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names)))


model_10 = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=10,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
model_50 = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=50,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
model_100 = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

model_10.fit(X_train, training_y)
y_10 = model_10.predict(X_test)
print("Score: {:.2f}".format(model_10.score(X_test, testing_y)))

model_50.fit(X_train, training_y)
y_50 = model_50.predict(X_test)
print("Score: {:.2f}".format(model_50.score(X_test, testing_y)))

model_100.fit(X_train, training_y)
y_100 = model_100.predict(X_test)
print("Score: {:.2f}".format(model_100.score(X_test, testing_y)))


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

precision = precision_score(testing_y, y_100, average='binary')
print('Precision 100 iter: %.3f' % precision)

recall = recall_score(testing_y, y_100, average='binary')
print('Recall 100 iter: %.3f' % recall)

score = f1_score(testing_y, y_100, average='binary')
print('F-Measure 100 iter: %.3f' % score)




precision = precision_score(testing_y, y_50, average='binary')
print('Precision 50 iter: %.3f' % precision)

recall = recall_score(testing_y, y_50, average='binary')
print('Recall 50 iter: %.3f' % recall)

score = f1_score(testing_y, y_50, average='binary')
print('F-Measure 50 iter: %.3f' % score)



precision = precision_score(testing_y, y_10, average='binary')
print('Precision 10 iter: %.3f' % precision)

recall = recall_score(testing_y, y_10, average='binary')
print('Recall 10 iter: %.3f' % recall)

score = f1_score(testing_y, y_10, average='binary')
print('F-Measure 10 iter: %.3f' % score)

'''
BEST output for iteration 100

Precision 100 iter: 0.877
Recall 100 iter: 0.880
F-Measure 100 iter: 0.878

'''

test_sample_1 = "This movie is was fantastic. I will go there again"
test_sample_2 = " Highly_recommended."
test_sample_3 = " May be I like this movie. Can't assure "
test_sample_4 = " I will think twice before recommending"
test_sample_5 = " Not to my taste. Will skip and watch another movie."
test_sample_6 = " The movie really sucks!"

test_samples = [test_sample_1,test_sample_2, test_sample_3,test_sample_4,test_sample_5,test_sample_6]

output = model_100.predict(vect.transform(test_samples))
print(str(output))
