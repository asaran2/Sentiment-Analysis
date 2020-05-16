# Sentiment-Analysis

Implemented a Naïve Bayes classifier in Java for categorizing movie reviews as either POSITIVE or NEGATIVE. 

The dataset consists of online movie reviews derived from an IMDb dataset: https://ai.stanford.edu/~amaas/data/sentiment/ that have been labeled based on the review scores. A negative review has a score ≤ 4 out of 10, and a positive review has a score ≥ 7 out of 10. We have done some preprocessing on the original dataset to remove some noisy features. Each row in the training set and test set files contains one review, where the first word in each line is the class label (1 represents POSITIVE and 0 represents NEGATIVE) and the remainder of the line is the review text.

Testing:
Use the following command line format:
java SentimentAnalysis mode trainFilename [testFilename | K]
  
where trainingFilename and testFilename are the names of the training set and test set files, respectively. mode is an integer from 0 to 3, controlling what the program will output. When mode is 0 or 1, there are only two arguments, mode and trainFilename; when the mode is 2 the third argument is testFilename; when mode is 3, the third argument is K, the number of folds used for cross validation. The output for these four modes should be:
  
0. Prints the number of documents for each label in the training set
1. Prints the number of words for each label in the training set
2. For each instance in test set, prints a line displaying the predicted class and the log probabilities for both classes
3. Prints the accuracy score for K-fold cross validation

(sample test files and results provided)
