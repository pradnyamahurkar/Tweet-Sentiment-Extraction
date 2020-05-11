import pandas as pd

# Data filepaths
TRAIN_FILEPATH = 'input/train.csv'
TEST_FILEPATH = 'input/test.csv'
SUBMISSION_FILEPATH = 'output/submission.csv'


# Given two strings, compute their Jaccard index.
def jaccard(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


# Given an array of tweets and an array of selections to validate with, compute
# the average Jaccard score for the whole set.
def validate(tweets, selects):
    n = len(tweets)
    
    total = 0
    for i in range(n):
        total += jaccard(str(tweets[i]), str(selects[i]))
    
    return total / n


def main():
    # Read validation data
    csv_tr = pd.read_csv(TRAIN_FILEPATH)
    tweet_va = csv_tr.iloc[:, 1]  # Full tweets
    select_va = csv_tr.iloc[:, 2]  # Selected texts
    
    # Read testing data
    csv_te = pd.read_csv(TEST_FILEPATH)
    id_te = csv_te.iloc[:, 0]  # Tweet IDs
    tweet_te = csv_te.iloc[:, 1]  # Full tweets
    
    # Get the validation score when using the whole tweet as the prediction
    validation_score = validate(tweet_va, select_va)
    print('Validation score: {0:0.5f}'.format(validation_score))
    
    # Create a submission for the test data in this fashion
    submission = {'textID': list(id_te), 'selected_text': list(tweet_te)}
    pd.DataFrame(submission).to_csv(SUBMISSION_FILEPATH, index=False)


if __name__ == '__main__':
    main()
