### Sentiment analysis over IMDB movie review
Goal is to predict sentiment from review.
###Getting started
`pip install -r reqirements.txt`

####Dataset
Dataset for embedding: https://www.kaggle.com/nltkdata/movie-review

Dataset for training: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

####Problem statement
From IMDB dataset use 90% data for training. Train three Logistic regression models with three different iterations:10,50 and 100. Validate these models on remaining 10% of the given data and choose the model with best accuracy. Use the best model to predict labels of the 10% remaining data that was used for validation. Once you get the predicted labels, calculate the precision, recall and F-measure. 