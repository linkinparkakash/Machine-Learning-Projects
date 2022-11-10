import csv
import sklearn
import pickle
import pandas as pd
import numpy as np
import nltk
import string
import re
from nltk.stem.porter import PorterStemmer
nltk.download("punkt")
import warnings
nltk.download('stopwords')
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import logging as log

class preprocessing_and_training:
    """
    Description: This class will preprocess the dataset and train the model for predictions.
    Output: It will create two pickle files, one for the vectors and other one is the machine learning model.
    Version: 1.0
    Revision: None
    """

    try:
            def __init__(self, file_path):
                self.file_path = file_path

            def start(self):
                """
                Description: Calling this method will start the preprocessing and training the model.
                Version: 1.0 
                Revision: None

                """
                df = pd.read_csv("dataset_spam.csv")

                # Removing all the punctuation, removing stopwords, performing stemming, lemmatization, and converting the text into vectors.
                corpus = []
                length = len(df)
                for i in range(0,length):
                    text = re.sub("[^a-zA-Z0-9]"," ",df["sample_text"][i])
                    text = text.lower()
                    text = text.split()
                    pe = PorterStemmer()
                    stopword = stopwords.words("english")
                    text = [pe.stem(word) for word in text if not word in set(stopword)]
                    text = " ".join(text)
                    corpus.append(text)
                
                cv = CountVectorizer(max_features=35000)
                X = cv.fit_transform(corpus).toarray()

                # Extracting dependent variable from the dataset
                y = pd.get_dummies(df['output_str'])
                y = y.iloc[:, 1].values

                ## saving to into cv.pkl file
                pickle_1 = pickle.dump(cv, open('cv.pkl', 'wb'))

                # Splitting the training and test sets from the dataset in the ususal 80:20 ratio.
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
                
                # Algorithm select and fitting the features and target columns.
                classifier = MultinomialNB(alpha=0.3)
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                y_pred

                # Saving the model that will be responsible for the training at the end.
                pikle_2 = pickle.dump(classifier, open("spam.pkl", "wb"))   

                return pickle_1
                return pickle_2           

    except Exception as e:
        log.warning('An error has occurred: {}'.format(e))
        raise Exception
        
#training = preprocessing_and_training('/config/workspace/dataset_spam.csv')
#training.start()