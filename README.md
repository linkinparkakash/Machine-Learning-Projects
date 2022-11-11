# Spam SMS Classifier.

The increased number of unsolicited emails known as spam has necessitated the
development of increasingly reliable and robust antispam filters. Recent machine learning
approaches have been successful in detecting and filtering spam emails.

This model aims to predict whether an email is spam or not spam (ham). The dataset consists of email messages and their labels (0 for ham, 1 for spam).

# Dataset

Link - http://www.grumbletext.co.uk/

It contains one set of SMS messages in English of 5,574 messages, tagged according being ham (legitimate) or spam. 14.4% of these sample texts are spam and rest are ham.

# Data Cleaning

The dataset came quite unclean, it didn't even have a proper .csv format, there was no column, everything was just into one big mess.
1. Gave the file a proper .csv format.
2. Separated the sapm\ham lables into different column and encoded them in numerical form.
3. Created three columns in total.

  sample_text - It contains the sample text messages.
  output_str - It contains the labels for each sample text, and has two values spam or ham
  output_num - It contains the encoded labels, 1 for spam and 0 for ham.
  
# Data Preprocessing

To preprocess the data, the sample_text was vectorized and then put to training.

# Training

The Naïve Bayes Multiple NOminal NB was used to train the model which gave around 98.68% accuracy.

# Model
You can test it here - https://hamspamclassifiernew.herokuapp.com/



