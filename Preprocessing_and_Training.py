import csv
import sklearn
import pickle
import pandas as pd
import numpy as np
import nltk
import nltk
import string
nltk.download("punkt")
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedKFold,cross_val_score,learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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

                data = pd.read_csv('dataset_spam.csv', encoding='latin-1')
                data = data.rename(columns={"sample_text" : "text", "output_str":"label"})

                data = data.replace(['ham','spam'],[0, 1])


                #remove the punctuations and stopwords
                def text_process(text):

                    text = text.translate(str.maketrans('', '', string.punctuation))
                    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

                    return " ".join(text)

                data['text'] = data['text'].apply(text_process)

                # Now, create a data frame from the processed data before moving to the next step.

                text = pd.DataFrame(data['text'])
                label = pd.DataFrame(data['label'])

                # Counting how many times a word appears in the dataset

                from collections import Counter

                total_counts = Counter()
                for i in range(len(text)):
                    for word in text.values[i][0].split(" "):
                        total_counts[word] += 1

                # Sorting in decreasing order (Word with highest frequency appears first)
                vocab = sorted(total_counts, key=total_counts.get, reverse=True)
                print(vocab[:60])

                # Mapping from words to index

                vocab_size = len(vocab)
                word2idx = {}
                #print vocab_size
                for i, word in enumerate(vocab):
                    word2idx[word] = i


                # Text to Vector
                def text_to_vector(text):
                    word_vector = np.zeros(vocab_size)
                    for word in text.split(" "):
                        if word2idx.get(word) is None:
                            continue
                        else:
                            word_vector[word2idx.get(word)] += 1
                    return np.array(word_vector)

               # Convert all titles to vectors
                word_vectors = np.zeros((len(text), len(vocab)), dtype=np.int_)
                for i, (_, text_) in enumerate(text.iterrows()):
                    word_vectors[i] = text_to_vector(text_[0])


                #convert the text data into vectors
                from sklearn.feature_extraction.text import TfidfVectorizer

                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform(data['text'])
                features = vectors
                                
                # saving to into cv.pkl file
                pickle.dump(vectors, open('vectors.pkl', 'wb'))

                #split the dataset into train and test set
                X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.15, random_state=111)

                mnb = MultinomialNB(alpha=0.2)

                clfs = {'NB': mnb}

                def train(clf, features, targets):
                    clf.fit(features, targets)

                def predict(clf, features):
                    return (clf.predict(features))
                
                # Saving the machine learnign model
                pickle.dump(mnb, open("spam_classifier_model.pkl", "wb"))

                pred_scores_word_vectors = []
                for k,v in clfs.items():
                    train(v, X_train, y_train)
                    pred = predict(v, X_test)
                    pred_scores_word_vectors.append((k, [accuracy_score(y_test , pred)]))

                    return pred_scores_word_vectors

    except Exception as e:
        self.log.warning('An error has occurred: {}'.format(e))
        raise Exception