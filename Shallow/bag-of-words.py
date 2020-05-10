import numpy as np
import pandas as pd
import re

# Data filepaths
TRAIN_FILEPATH = 'input/train.csv'
TEST_FILEPATH = 'input/test.csv'
HP_FILEPATH = 'output/hp_results.csv'
SUBMISSION_FILEPATH = 'output/submission.csv'

# Hyperparameter sets
EPSILONS = [0.5, 1]
BATCH_SIZES = [64, 128, 256, 512]
NUM_EPOCHS = [30, 40, 50]
ALPHAS = [0.00005, 0.0001, 0.0005]


# Given a string, remove bad characters and return a cleaned up string. If the
# string is NaN, return an empty string.
def clean(string):
    # If string is np.NaN, make it an empty string
    if string is np.nan:
        return ''
    
    string = string.lower()
    
    # Remove URLs and punctuation that should not separate words
    string = re.sub(r'[\'\"`]|http://[^\s]*', '', string)
    
    # Replace punctuation that should separate words with spaces
    string = re.sub('[.,!?(){};:]', ' ', string)
    
    return string


# Given an array of tweets and their sentiments, return a map of all the unique
# words within them to a unique ID number.
def create_vocab(tweets, sents):
    vocab = dict()
    word_id = 0
    
    for i in range(len(tweets)):
        
        # Skip words in neutral tweets
        if sents[i] == 'neutral':
            continue
        
        tweet = tweets[i]
        for word in tweet.split():
            # Check if new word
            if word not in vocab:
                # Add to vocab dictionary with a unique ID number
                vocab[word] = word_id
                word_id += 1
    
    return vocab


# Given a vocabulary dict (size m) and an array of all the selected
# phrases (1 x n), return a design matrix Xtilde (m x n) where each example
# is a column bag of words representation of the phrase.
def create_Xtilde(vocab, selections):
    m = len(vocab)
    n = len(selections)
    Xtilde = np.zeros((m, n))
    
    for i in range(n):
        # Count all words in this selection, leave absent words as 0
        for word in selections[i].split():
            if word in vocab:
                Xtilde[vocab[word], i] += 1
    
    # Stack bias term on bottom of matrix
    Xtilde = np.vstack((Xtilde, np.ones(n)))
    
    return Xtilde


# Given a vector of weights w, a design matrix Xtilde, and a vector of
# labels y, return the MSE.
def fMSE(w, Xtilde, y):
    diffs = Xtilde.T.dot(w) - y  # Difference of guess and ground-truth
    n = len(y)
    return (0.5 / n) * np.sum(diffs ** 2)


# Given a vector of weights w, a design matrix Xtilde, and a vector of
# labels y, return the gradient of the MSE loss.
def gradfMSE(w, Xtilde, y, alpha):
    L2 = alpha * w
    return (1 / len(y)) * Xtilde.dot((Xtilde.T.dot(w) - y)) + L2


# Given a design matrix Xtilde (m x n) and labels y (1 x n), train a linear
# regressor for Xtilde and y using gradient descent on fMSE.
def SGD(Xtilde, y, hp, verbose=False):
    epsilon, batches, epochs, alpha = hp
    
    # Number of examples
    n = len(y)
    
    # Shuffle X and Y to create mini-batches with
    randOrder = np.arange(n)
    np.random.shuffle(randOrder)
    X_shuffled = Xtilde[:, randOrder]
    y_shuffled = y[randOrder]
    
    # Initialize w with normal distribution
    w = 0.005 * np.random.randn(Xtilde.shape[0])
    
    # Perform SGD
    if verbose:
        print('Epoch     MSE Loss')
    for e in range(epochs):
        batch_size = n / batches
        for b in range(batches):
            # Create batch
            start = int(b * batch_size)
            end = int((b + 1) * batch_size)
            
            if b < batches - 1:
                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[start:end]
            else:
                # Don't forget the last example, go to the end
                X_batch = X_shuffled[:, start:]
                y_batch = y_shuffled[start:]
            
            # Learn!
            w = w - epsilon * gradfMSE(w, X_batch, y_batch, alpha)
        
        # Print some running results
        if verbose:
            loss = fMSE(w, Xtilde, y)
            print('{0:5d}   {1:12.8f}'.format(e + 1, loss))
    
    return w


# Given a vocab dictionary, a vector of word weights, a tweet, and a sentiment,
# return a substring from that tweet that best represents the sentiment.
def select_words(vocab, w, tweet, sent):
    # Neutral sentiment, select whole tweet
    if sent == 'neutral':
        return tweet
    
    # Split up words to evaluate one at a time
    words = tweet.split()
    
    # Kadane's algorithm (Find subarray of words with maximum sum value)
    local_max = 0
    global_max = 0
    start = 0
    end = len(words) - 1
    for i in range(len(words)):
        # Get word value
        word = clean(words[i])
        if word not in vocab:
            value = 0
        else:
            value = w[vocab[word]]
        
        if sent == 'negative':
            # Minimize the value if the sentiment is negative
            value *= -1
        
        local_max += value
        if value >= local_max:
            local_max = value
            start = i
        if local_max >= global_max:
            global_max = local_max
            end = i
    
    # Select words
    selection = ''
    for i in range(start, end + 1):
        selection += words[i] + ' '
    
    # Nothing good? Just return the whole tweet
    if selection == '':
        return tweet
    
    return selection[:-1]


# Given word info and a set of data to validate with, compute the average
# Jaccard score provided by the words selected by the model.
def validate(vocab, w, tweets, selects, sents):
    n = len(tweets)
    
    total = 0
    for i in range(n):
        true_selected = selects[i]
        pred_selected = select_words(vocab, w, tweets[i], sents[i])
        total += jaccard(true_selected, pred_selected)
    
    return total / n


def findBestHyper(Xtilde, y, vocab, tweet_va, select_va, sent_va):
    num_hps = len(EPSILONS) * len(BATCH_SIZES) * len(NUM_EPOCHS) * len(ALPHAS)
    i = 0
    
    best_score = 0
    best_hp = None
    results = np.zeros((num_hps, 5))
    for epsilon in EPSILONS:
        for batch_size in BATCH_SIZES:
            for num_epoch in NUM_EPOCHS:
                for alpha in ALPHAS:
                    print('\nBeginning SGD with hyperparameter set '
                          '{} of {}'.format(i + 1, num_hps))
                    print('Epsilon:\t', epsilon)
                    print('Batch Size:\t', batch_size)
                    print('Epochs:\t\t', num_epoch)
                    print('Alpha:\t\t', alpha)
                    
                    # Use stochastic gradient descent to train w
                    hp = (epsilon, batch_size, num_epoch, alpha)
                    
                    # Calculate the weights of the words
                    w = SGD(Xtilde, y, hp)
                    
                    # Get validation jaccard score using these weights
                    score = validate(vocab, w, tweet_va, select_va, sent_va)
                    if score > best_score:
                        best_score = score
                        best_hp = hp
                    
                    print('Validation set jaccard score: {0:.5f}'
                          .format(score))
                    
                    # Record results for this HP set
                    results[i] = np.array((*hp, score))
                    i += 1
                    break
                break
            break
        break
    
    # Output hyperparameter testing results to CSV
    headers = 'Epsilon, Batch Size, Epochs, Alpha, Jaccard'
    fmt = '%.2f, %d, %d, %.5f, %.4f'
    np.savetxt(HP_FILEPATH, results, fmt=fmt, header=headers, comments='')

    print('\nBest hyperparameters:', best_hp)
    print('Best Jaccard Score: {0:.5f}\n'.format(best_score))
    return best_hp


# Given word info and tweets along with their IDs and sentiments, return a
# pandas DataFrame containing the properly formatted submission data.
def create_submission(vocab, w, ids, tweets, sents):
    # Create submission data
    submission = {'textID': list(ids), 'selected_text': list()}
    
    # Select words in the testing set
    for i in range(len(tweets)):
        selection = select_words(vocab, w, tweets[i], sents[i])
        submission['selected_text'].append(selection)
    
    # Output final submissions
    return pd.DataFrame(submission)


# Given two strings, compute their Jaccard index.
def jaccard(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def main():
    VALID_SIZE = 5000
    
    # Read training data
    csv_tr = pd.read_csv(TRAIN_FILEPATH)
    tweet_tr = np.array(csv_tr.iloc[:-VALID_SIZE, 1])  # Full tweets
    select_tr = np.array(csv_tr.iloc[:-VALID_SIZE, 2])  # Selected texts
    sent_tr = np.array(csv_tr.iloc[:-VALID_SIZE, 3])  # Sentiments
    
    # Read validation data
    tweet_va = np.array(csv_tr.iloc[VALID_SIZE:, 1])  # Full tweets
    select_va = np.array(csv_tr.iloc[VALID_SIZE:, 2])  # Selected texts
    sent_va = np.array(csv_tr.iloc[VALID_SIZE:, 3])  # Sentiments
    
    # Read testing data
    csv_te = pd.read_csv(TEST_FILEPATH)
    id_te = np.array(csv_te.iloc[:, 0])  # Tweet IDs
    tweet_te = np.array(csv_te.iloc[:, 1])  # Full tweets
    sent_te = np.array(csv_te.iloc[:, 2])  # Sentiments
    
    # Clean up training tweets
    for i in range(len(tweet_tr)):
        tweet_tr[i] = clean(tweet_tr[i])
    
    for i in range(len(select_tr)):
        select_tr[i] = clean(select_tr[i])
    
    # Get full vocabulary of words from all cleaned tweets
    vocab = create_vocab(tweet_tr, sent_tr)
    
    # Create design matrix Xtilde
    Xtilde = create_Xtilde(vocab, select_tr)
    
    # Create labels vector y
    n = len(sent_tr)
    y = np.zeros(n)
    for i in range(n):
        if sent_tr[i] is 'positive':
            y[i] = 1
        elif sent_tr[i] is 'negative':
            y[i] = -1
    
    # Find the best set of hyperparameters
    best_hp = findBestHyper(Xtilde, y, vocab, tweet_va, select_va, sent_va)
    
    # Train a model using them
    w = SGD(Xtilde, y, best_hp, verbose=True)
    
    # Create submission with best weight vector
    submission_df = create_submission(vocab, w, id_te, tweet_te, sent_te)
    submission_df.to_csv(SUBMISSION_FILEPATH, index=False)


if __name__ == '__main__':
    main()
