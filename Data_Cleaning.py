# This is just a stand alone module, to show what has been done to the actual dataset, how it was cleaned.
# Note: Further changes will be made to the dataset, column names will be changed to suit the training process.
# This module is no longer needed, only for reading purposes, please check the EDA jupyternotebook file for more details.


import pandas as pd 
import csv

# Loading the dataste.
df = pd.read_csv('/config/workspace/Spam_Ham_Classifier/dataset.csv')

# Appending the columns in the dataset.
with open('/config/workspace/Spam_Ham_Classifier/dataset.csv') as file_obj:
    col = []
    heading = next(file_obj)
    reader_obj = csv.reader(file_obj)
    for row in reader_obj:
        col.append(row)

# Defining the new transformed dataframe.
df1 = pd.DataFrame(col)

#Giving name to the column
df1['sample_text'] = df1[0]
df1.drop([0],axis = 1, inplace = True)

# Making two columns from sample_text column
pd.DataFrame(df1.sample_text.str.split('/t'))
df2 = pd.concat([df,df1], axis = 1)

# Renaming the column  
df2.rename(columns = {'ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...':'output_str'}, inplace = True)

# Splitting the column characters to extract spam and ham outputs
df3 = df2.output_str.str.split(expand=True)

# We only need the 0th column
df4 = df3[0]

# Saving it into a .csv file, later will be added with the df2
df4.to_csv('/config/workspace/Spam_Ham_Classifier/dataset.csv')
df4 = pd.read_csv('/config/workspace/Spam_Ham_Classifier/dataset.csv')


# Renaming the column and appending it into the df2
df4.rename(columns = {'0':'output_str'}, inplace = True)
df5 = pd.concat([df2, df4], axis=1)
df5.drop(columns = 'output_str', inplace = True)

# Removing another unwanted column
df5.drop(columns = 'Unnamed: 0', inplace = True)
df5

df5

# Renaming the output column
df5.rename(columns = {'Unnamed: 0.1':'output_str'}, inplace = True)


# Numbering the output string values.
# '0' means ham, and '1' means spam

output= []
for i in df5['output_str']:
    if i == 'ham':
        output.append(0)
    else:
        output.append(1)

# Making it a DataFrame file now
df6 = pd.DataFrame(output)

# Renaming the column name
df6.rename(columns = {0:'output_num'}, inplace = True)

# Now appending it back in df5
df7 = pd.concat([df5, df6], axis=1)

# Splitting the spam and ham from the sample_text column
df7['sample_text'] = df7['sample_text'].map(lambda x: x.lstrip('ham\t'))
df7['sample_text'] = df7['sample_text'].map(lambda x: x.lstrip('spam\t'))
df7.drop(columns = 'index', inplace = True)


# Saving the cleaned dataset in a new .csv file
df7.to_csv('/config/workspace/Spam_Ham_Classifier/dataset.csv')