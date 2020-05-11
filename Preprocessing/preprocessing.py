# Preprocessing code to get initial data view and exploration of the train.csv file
# This is specifically for the Kaggle Twitter sentiment analysis project, but can be adapted for any dataset.
# by svadivazhagu, May 2020

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#Load the data in
def loadData(fp="../test.csv"):
    train = pd.read_csv(fp)
    return train

#Text-based information about the dataset
def dataStats(train):
    #Get an example viewing of what some of the data looks like + num columns
    train.head()

    #Show how many samples there are as well as num attributes
    print(f'{train.shape[0]} tweets, with {train.shape[1]} attributes')

    #Show how many tweets exist per sentiment category
    train['sentiment'].value_counts()

    #Average the length of the tweets for each category
    neutral, positive, negative = int((train.loc[train['sentiment'] == 'neutral'])['text'].str.len().mean()), \
                                  int((train.loc[train['sentiment'] == 'positive'])['text'].str.len().mean()),\
                                  int((train.loc[train['sentiment'] == 'negative'])['text'].str.len().mean())

    print(f' On average, {neutral} chars. per neutral tweet, {positive} per positive, and {negative} per negative')

def visualStats(train):

    #count tweets per sentiment category
    plt.figure(figsize=(7,5))
    plt.title('Tweets per sentiment category')
    sns.countplot(train['sentiment'])
    plt.show()

    #check the distribution of each category's tweet length
    neutral = train[train['sentiment'] == 'neutral']['text'].apply(lambda x: len(str(x)))
    positive = train[train['sentiment'] == 'positive']['text'].apply(lambda x: len(str(x)))
    negative = train[train['sentiment'] == 'negative']['text'].apply(lambda x: len(str(x)))
    fig, ax = plt.subplots()
    sns.distplot(neutral, ax=ax, color='green')
    sns.distplot(positive, ax=ax, color='blue')
    sns.distplot(negative, ax=ax, color='red')
    plt.title('Length of tweet based on sentiment')
    plt.legend(['neutral', 'positive', 'negative'])
    plt.show()

def cleanText(tweet):
    #list of emoji patterns appearing in the tweets to be removed
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = str(tweet)
    # Remove emojis
    text = emoji_pattern.sub(r'', text)
    # Remove twitter handles (@___)
    text = re.sub(r'@\w+', '', text)
    # Remove links after research that t.co uses http still
    text = re.sub(r'http.?://[^/s]+[/s]?', '', text)
    return text.strip().lower()


def wordcloud(train, text='text'):
    # Join all tweets in one string
    corpus = " ".join(str(review) for review in train[text])
    print(f"There are {len(corpus)} words used.")

    wordcloud = WordCloud(max_font_size=50,
                          max_words=100,
                          background_color="white").generate(corpus)

    plt.figure(figsize=(30, 30))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    train = loadData()
    dataStats(train)
    visualStats(train)

    train['text'] = train['text'].apply(lambda x: cleanText(x))

    #Check the most common words across the cleaned tweets
    wordcloud(train=train, text = 'text')