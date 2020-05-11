# sentimentextraction
Tweet sentiment extraction NLP challenge as per https://www.kaggle.com/c/tweet-sentiment-extraction/

## svadivazhagu TODO
	- Shallow -Simple counting approach
	- Medium - bag of words/fasttext
	- Deep - Tf/PyTorch BERT/ Roberta.
	
	-Work on report

## insights after preprocessing
	What if the user links a gif in the tweet of a sad person and they make a sarcastic happy tweet? should class as negative
	What if the user puts a sad emoji in? if we had more time then we could add weights for emojis which would make the accuracy a lot higher maybe
	Could probably integrate hyperparameter optimization with this model to make it more fine-tuned for the data but either would take too much time or is too challenging given the limited amount of time left 


##Manual work on optimization
Different loss function - categorical cross entropy is awful for this -- loss of 35-40
