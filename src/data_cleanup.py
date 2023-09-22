import emoji
import nltk

import os.path

import pandas as pd

# Download stopwords from nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

# Since the dataframe is being altered (e.g. for dropna, encoding emoji), the warning is being disabled
pd.options.mode.chained_assignment = None
words = set(stopwords.words('english'))


def process_dataset(data, attr):
    print('Empty Row Count:', data[attr].isna().sum())
    data = drop_empty(data)
    print('Empty Row Count:', data[attr].isna().sum())

    print('Encoding Emoji')
    encode_emoji(data, attr)

    print('Removing stopwords')
    remove_stopwords(data, attr)

    print('Cleaning-up Unwanted Symbols')
    data_cleanup(data, attr)

    print('Removing potential duplicates')
    data = data.drop_duplicates()

    return data


def drop_empty(data):
    print('Dropping Empty Rows')
    data = data.dropna()
    data = data.reset_index()
    return data


def encode_emoji(data, attr):
    for idx in range(0, len(data)):
        data[attr][idx] = ''.join(' ' + e + ' ' if emoji.is_emoji(e) else e for e in data[attr][idx])


def data_cleanup(data, attr):
    data[attr] = data[attr].str.replace('\n', ' ')
    data[attr] = data[attr].str.replace('(http|https)[\\S]+', '', regex=True)
    data[attr] = data[attr].str.replace('(@\\S+)', '', regex=True)
    data[attr] = data[attr].str.replace('[^a-zA-Z\\s]', '', regex=True)
    data[attr] = data[attr].str.lower()


def remove_stopwords(df, attr):
    df[attr] = df[attr].apply(lambda t: ' '.join([w for w in t.split() if w not in words]))


print('Reading training data from `data` directory')
train_data = pd.read_csv('../data/train.En.csv')
print('Row Count:', len(train_data))

print('Reading test data for Task A from `data` directory')
test_data_a = pd.read_csv('../data/task_A_En_test.csv')
print('Row Count:', len(test_data_a))

print('Reading test data for Task A from `data` directory')
test_data_b = pd.read_csv('../data/task_B_En_test.csv')
print('Row Count:', len(test_data_b))

print("=================================================")

print('Processing Train Data for Task A')
train_data_a = train_data[['tweet', 'sarcastic']]
train_data_a = process_dataset(train_data_a, 'tweet')

print("=================================================")

print('Processing Test Data for Task A')
processed_test_data_a = test_data_a[['text', 'sarcastic']]
processed_test_data_a.rename(columns={'text': 'tweet'}, inplace=True)
processed_test_data_a = process_dataset(processed_test_data_a, 'tweet')

print("=================================================")

print('Process Train Data for Task B')
train_data_b = train_data[
    ['rephrase', 'sarcasm', 'irony', 'satire', 'understatement', 'overstatement', 'rhetorical_question']]

train_data_b = process_dataset(train_data_b, 'rephrase')

print("=================================================")

print('Processing Test Data for Task B')
processed_test_data_b = test_data_b[
    ['text', 'sarcasm', 'irony', 'satire', 'understatement', 'overstatement', 'rhetorical_question']]
processed_test_data_b.rename(columns={'text': 'rephrase'}, inplace=True)
processed_test_data_b = process_dataset(processed_test_data_b, 'rephrase')

print("=================================================")

print('Writing cleaned-up data set in `clean_data` directory')
if not os.path.exists('../clean_data'):
    os.mkdir('../clean_data')

train_data_a.to_csv('../clean_data/train_task_a.csv', index=False)
processed_test_data_a.to_csv('../clean_data/test_task_a.csv', index=False)

train_data_b.to_csv('../clean_data/train_task_b.csv', index=False)
processed_test_data_b.to_csv('../clean_data/test_task_b.csv', index=False)
