### Sentiment analysis over IMDB movie review
Goal is to predict sentiment from review.
### Getting started
`pip install -r reqirements.txt`

#### Dataset

Dataset for embedding: https://www.kaggle.com/nltkdata/movie-review

Dataset for training: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

#### Problem statement
###### Goal 1: 
From IMDB dataset use 90% data for training. Train three Logistic regression models with three different iterations:10,50 and 100. Validate these models on remaining 10% of the given data and choose the model with best accuracy. Use the best model to predict labels of the 10% remaining data that was used for validation. Once you get the predicted labels, calculate the precision, recall and F-measure. 

###### Goal 2: 
Train an embedding model on the nltkdata dataset and use this embedding to train the following neural network:
Try to use a dataframe other than pandas to load the training data and filter those rows having less than 150 words. While pre-processing show how did you consider padding the data.Take 90% of the given dataset(IMDB dataset) and train a neural network with maximum of three layers where the first layer will be an embedding layer(trained on nltkdata). Then use this model to predict labels of the remaining 10% data and report precision,recall and F-measure.

