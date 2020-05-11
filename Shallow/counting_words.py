
"""
Created on Wed Apr 29 01:48:30 2020

@author: pradnyamahurkar
"""

import pandas as pd 
import numpy as np

# CountVectorizer will help calculate word counts
from sklearn.feature_extraction.text import CountVectorizer

# Make training/test split
from sklearn.model_selection import train_test_split

# Import the string dictionary that we'll use to remove punctuation
import string

# Import datasets
X_tr = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
X_tr.text = X_tr.text.astype(str)
X_te = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
X_te.text = X_te.text.astype(str)
# sample = pd.read_csv('sample_submission.csv')

# Make all the text lowercase - casing doesn't matter when 
# we choose our selected text.
X_tr['text'] = X_tr['text'].apply(lambda x: x.lower())
X_te['text'] = X_te['text'].apply(lambda x: x.lower())

X_train, X_test = train_test_split(X_tr, train_size = 0.5, random_state = 0)

pos_tr = X_train[X_train['sentiment'] == 'positive']
neutral_tr = X_train[X_train['sentiment'] == 'neutral']
neg_tr = X_train[X_train['sentiment'] == 'negative']

# dictionary of words
pos_words = {}
neutral_words = {}
neg_words = {}

# substracting words which are used for other sentiments
neg_words_acc = {}
pos_words_acc = {}
neutral_words_acc = {}

# Use CountVectorizer to get the word counts within each dataset

cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, stop_words='english')

X_train_cv = cv.fit_transform(X_train['text'])

X_pos = cv.transform(pos_tr['text'])
X_neutral = cv.transform(neutral_tr['text'])
X_neg = cv.transform(neg_tr['text'])

pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())

for k in cv.get_feature_names():
    pos = pos_count_df[k].sum()
    neutral = neutral_count_df[k].sum()
    neg = neg_count_df[k].sum()
    
    pos_words[k] = pos/pos_tr.shape[0]
    neutral_words[k] = neutral/neutral_tr.shape[0]
    neg_words[k] = neg/neg_tr.shape[0]

for key, value in neg_words.items():
    neg_words_acc[key] = neg_words[key] - (neutral_words[key] + pos_words[key])
    
for key, value in pos_words.items():
    pos_words_acc[key] = pos_words[key] - (neutral_words[key] + neg_words[key])
    
for key, value in neutral_words.items():
    neutral_words_acc[key] = neutral_words[key] - (neg_words[key] + pos_words[key])
    
def calculate_selected_text(df_row, tol = 0):
    
    tweet = df_row['text']
    sentiment = df_row['sentiment']
    
    if(sentiment == 'neutral'):
        return " ".join(str(tweet).lower().split())
    elif(sentiment == 'positive'):
        dict_to_use = pos_words_acc 
    elif(sentiment == 'negative'):
        dict_to_use = neg_words_acc 
        
    words = tweet.split()
    words_len = len(words)
    subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]
    
    score = 0
    selection_str = '' 
    sortedlist = sorted(subsets, key = len) 
    
    for i in range(len(subsets)):
        
        new_sum = 0 # Sum for the current substring
        
        # Calculate the sum of weights for each word in the substring
        for p in range(len(sortedlist[i])):
            if(sortedlist[i][p].translate(str.maketrans('','',string.punctuation)) in dict_to_use.keys()):
                new_sum += dict_to_use[sortedlist[i][p].translate(str.maketrans('','',string.punctuation))]
            
        if(new_sum > score + tol):
            score = new_sum
            selection_str = sortedlist[i]
            tol = tol*2 # Increase the tolerance a bit each time we choose a selection

    # If we didn't find good substrings, return the whole text
    if(len(selection_str) == 0):
        selection_str = words
        
    return ' '.join(selection_str)


tol = 0.001

X_test['predicted_selection'] = ''

for index, row in X_test.iterrows():
    selected_text = calculate_selected_text(row, tol)
    X_test.loc[X_test['textID'] == row['textID'], ['predicted_selection']] = selected_text
    
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

 # validation set
X_test['jaccard'] = X_test.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)
print('The jaccard score for the validation set is:', np.mean(X_test['jaccard']))

# test set
X_te['selected_text'] = ''

for index, row in X_te.iterrows():
    selected_text = calculate_selected_text(row, tol)
    X_te.loc[X_te['textID'] == row['textID'], ['selected_text']] = selected_text

X_te.drop(columns=['text', 'sentiment'])

X_te.to_csv("submission.csv", index=False)
X_te.sample(10)
#print('The jaccard score for the validation set is:', np.mean(X_test['jaccard']))
