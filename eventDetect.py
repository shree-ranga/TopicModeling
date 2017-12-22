# Event Detection from Tweets
# -*- coding: utf-8 -*-

__author__ = 'Shree Ranga Raju'

# Import Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

import re
from collections import Counter

import fastcluster
import CMUTweetTagger

import scipy.cluster.hierarchy as sch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import metrics




# Load stop words from nltk library
def load_stopwords():
	stop_words = nltk.corpus.stopwords.words('english')
	stop_words.extend(['rt&amp', '&amp', 'rt', 'retweet'])
	stop_words = set(stop_words)
	return stop_words

# Normalize text to remove urls, usermentions, hashtags, digits and other punctuations.
# Boiler plate code available for normalizing text at github.com/heerme.
def normalize_text(text):
	try:
		text = text.encode('utf-8')
	except: pass
	# The below code only works for ucs4. Mine is ucs2 :(
	# myre = re.compile(u'['u'-\U0001F300\U0001F64F'
	# 				  u'\U0001F680\U0001F6FF'']+')
	# text = myre.sub('', text)
	text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', text)
	text = re.sub('@[^\s]+','', text)
	text = re.sub('#([^\s]+)', '', text)
	text = re.sub('[:;>?<=*+()/,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]','', text)
	text = re.sub('[\d]','', text)
	text = text.replace(".", '')
	text = text.replace("'", '')
	text = text.replace("\"", '')
	# Normalize some utf8 encoding of special characters (emojis) if exists. Below is an example.
	# text = text.replace("\xb9",' ').replace("\xba",' ')
	text = text.replace("\xc2\xa3m\xa6", ' ')
	text = text.replace("\xa6", ' ')
	text = text.replace("\xad", ' ')
	text = text.replace("\xb2", ' ')
	text = text.replace("\x94", ' ')
	text = text.replace("\xa4", ' ')
	text = text.replace("\xa3", ' ')
	text = text.replace("\xab", ' ')
	text = text.replace("\xac", ' ')
	text = text.replace("\xb4", ' ')
	text = text.replace("\xc3\xa9", ' ')
	text = text.replace("\x85\x86\x86\x86\x86\xa6", ' ')
	text = text.replace("\x9d", ' ')
	text = text.replace("\xd8", ' ')
	text = text.replace("\xf0\x9f\x8c\xb9\xf0\x9f\x8c\xb9\xf0\x9f\x8c\xb9", ' ')
	text = text.replace("\xa0", ' ')
	text = text.replace("\xb0", ' ')
	text = text.replace("\xa7", ' ')
	text = text.replace("\xb1", ' ')
	text = text.replace("\xb3", ' ')
	text = text.replace("\xb9", ' ')
	text = text.replace("\xaa", ' ')
	text = text.replace("\xae", ' ')
	text = text.replace("\xc2", ' ')
	# text = text.replace("\U0001f3c8", ' ')
	# text = text.replace("\U0001f6a8", ' ')
	# text = text.replace("\U0001f44f", ' ')
	# text = text.replace("\U0001f389\U0001f38a\U0001f451", ' ')
	# text = text.replace("\U0001f1fa\U0001f1f8", ' ')
	# text = text.replace("\u1f34b", ' ')
	return text

# Text tokenizer.
def tokenizer(text):
	features = []
	tokens = text.split()
	for word in tokens:
		# convert all the words to lower case
		# and also check if the length of the word is greater than 2
		if word.lower() not in stop_words and len(word) > 2:
			features.append(word)
	return features

# Text processor
def text_processor(text):
	text = normalize_text(text)
	features = tokenizer(text)
	return features


if __name__ == '__main__':

	debug = 0
	stop_words = load_stopwords()
	n_tweets = 0
	tid_to_raw_tweet = {} # tweet id to raw tweet
	tid_corpus = [] # tweet id corpus
	corpus = [] # tweet corpus

	# Database initialization
	client = pymongo.MongoClient('localhost:27017')
	if 'MyTweetsdb' not in client.database_names():
		print 'Database MyTweetsdb does not exist! Run getData.py.'

	for i in client.MyTweetsdb.tweets.find():
		text = i['tweet_text']
		features = text_processor(text)
		tweet_bag = ""
		n_tweets += 1
		# Make sure features has more than 3 tokens. Tweets are more meaningful that way.
		if len(features) > 3:
			for feature in features:
				tweet_bag += feature.decode('utf-8','ignore') + ","
			tweet_bag = tweet_bag[:-1]
			tid_corpus.append(i['tweet_id'])
			tid_to_raw_tweet[i['tweet_id']] = text
			corpus.append(tweet_bag)

	# Vectorizer
	# Minimum doc freq is 5. n-grams have to be present in at least 5 tweets to be considered as a topic
	# Binary is set to true for easier calculations and analysis
	# Minimum n-gram range is 2 and maximum is 3
	vectorizer = CountVectorizer(min_df = 5, binary = True, ngram_range = (2,3))
	X = vectorizer.fit_transform(corpus)

	# Get Vocabulary list
	vocX = vectorizer.get_feature_names()

	# More filtering of tweets based on vocabulary.
	# So much filtering because it helps in scaling.
	map_index_after_cleaning = {}
	Xclean = np.zeros((1, X.shape[1]))
	for i in range(0, X.shape[0]):
		if X[i].sum() >= 3	:
			Xclean = np.vstack([Xclean, X[i].toarray()])
			map_index_after_cleaning[Xclean.shape[0] - 2] = i

	Xclean = Xclean[1:,]

	print 'Total number of tweets in the database is {}'.format(n_tweets)
	print 'Total number of tweets in the corpus after first step of cleaning is {}'.format(len(corpus))
	print 'length of vocabulary is {}'.format(len(vocX))
	print 'Shape of X before vocabulary cleaning is {}'.format(X.shape)
	print 'Shape of X after vocabulary cleaning (Xclean) is {}'.format(Xclean.shape)
	print 'There are at most {} significant tweets among total number of ({}) tweets'.format(Xclean.shape[0], n_tweets)

	# Change the original X to Xclean
	X = Xclean

	# Scale the data to zero mean and unit variance. Also means calculate z-score. Standardization
	# Normalize the standardized using l_2 norm
	Xdense = np.matrix(X).astype('float')
	X_scaled = preprocessing.scale(Xdense)
	X_normalized = preprocessing.normalize(X_scaled, 'l2')

	# Distance Matrix ie sample by sample distances
	distMatrix = pairwise_distances(X_normalized, metric = 'cosine')

	print 'Fastcluster, cosine distance, average method'
	L = fastcluster.linkage(distMatrix, method = 'average')

	# Dendogram cutting threshold
	dt = 0.7

	indL = sch.fcluster(L, dt*distMatrix.max(), 'distance')

	freqTwCl = Counter(indL)

	print "n_clusters:", len(freqTwCl)

	print freqTwCl.most_common()







