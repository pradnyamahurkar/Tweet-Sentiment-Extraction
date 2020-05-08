#test of fasttext
#svadivazhagu, 2020
import pandas as pd
from gensim.models import FastText

#first import the train.csv and take a look atit
train = pd.read_csv("train.csv")
print(train.shape, train.sentiment.value_counts())

corpus = []

for i in train['text'].values:
    corpus.append(str(i).split(" "))

model = FastText(corpus, size=100, workers=4, window=5)


print(model.wv.most_similar('bad'))
print('******')
print(model.wv.most_similar('good'))