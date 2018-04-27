# Topic Detection from Tweets
# -*- coding: utf-8 -*-

__author__ = 'Shree Ranga Raju'

# Import Modules
import numpy as np 	
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
	# stop_words = nltk.download('stopwords')
	stop_words = nltk.corpus.stopwords.words('english')
	stop_words.extend(['rt&amp', '&amp', 'rt', 'retweet'])
	stop_words = set(stop_words)
	return stop_words

# Normalize text to remove urls, usermentions, hashtags, digits and other punctuations.
def normalize_text(text):
	try:
		text = text.encode('utf-8')
	except: pass
	text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', text)
	text = re.sub('@[^\s]+','', text)
	text = re.sub('#([^\s]+)', '', text)
	text = re.sub('[:;>?<=*+()/,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]','', text)
	text = re.sub('[\d]','', text)
	text = text.replace(".", '')
	text = text.replace("'", '')
	text = text.replace("\"", '')
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

	fdata = open("data.txt", "r")
	fout = open("output.txt", "w")
	stop_words = load_stopwords()
	tid = 0
	n_tweets = 0
	tid_to_raw_tweet = {} # tweet id to raw tweet
	tid_corpus = [] # tweet id corpus
	corpus = [] # tweet corpus

	debug = 1

	for i in fdata.readlines():
		text = i
		features = text_processor(text)
		tweet_bag = ""
		n_tweets += 1
		# Make sure features has more than 3 tokens. Tweets are more meaningful that way.
		if len(features) > 3:
			for feature in features:
				tweet_bag += feature.decode('utf-8','ignore') + ","
			tweet_bag = tweet_bag[:-1]
			tid_corpus.append(tid)
			tid_to_raw_tweet[tid] = text
			corpus.append(tweet_bag)
			tid += 1


	# Vectorizer
	# Minimum doc freq is 5. n-grams have to be present in at least 5 tweets to be considered as a topic
	# Binary is set to true for easier calculations and analysis
	# Minimum n-gram range is 2 and maximum is 3
	vectorizer = CountVectorizer(min_df = 5, binary = True, ngram_range = (2,3))
	X = vectorizer.fit_transform(corpus)

	# Get Vocabulary list
	vocX = vectorizer.get_feature_names()
	# print "Vocabulary: ", vocX
	# print "\n"

	# More filtering of tweets based on vocabulary.
	map_index_after_cleaning = {}
	Xclean = np.zeros((1, X.shape[1]))
	for i in range(0, X.shape[0]):
		if X[i].sum() >= 3	:
			Xclean = np.vstack([Xclean, X[i].toarray()])
			map_index_after_cleaning[Xclean.shape[0] - 2] = i

	Xclean = Xclean[1:,]

	# Change the original X to Xclean
	X = Xclean
	print "Shape of Xclean is {}".format(X.shape)
	print "The number of significant tweets is", X.shape[0]

	# POS Tokens (word, tag, confidence)
	boost_entity = {}
	pos_tokens = CMUTweetTagger.runtagger_parse([term.upper() for term in vocX])
	for l in pos_tokens:
		term = ''
		for gr in range(0,len(l)):
			term += l[gr][0].lower() + " "
		if "^" in str(l):
			boost_entity[term.strip()] = 2.5
		else:
			boost_entity[term.strip()] = 1.0

	# ranking?
	dfX = X.sum(axis=0)
	keys = vocX
	vals = dfX
	dfVoc = {}
	boosted_wdfVoc = {}
	# print "vals", vals
	for k,v in zip(keys, vals):
		dfVoc[k] = v
	for k in dfVoc:
		boosted_wdfVoc[k] = dfVoc[k] * boost_entity[k]
	# print sorted( ((v,k) for k,v in boosted_wdfVoc.iteritems()), reverse=True)


	# Scale the data to zero mean and unit variance.
	# Normalize the standardized using l_2 norm
	Xdense = np.matrix(X).astype('float')
	X_scaled = preprocessing.scale(Xdense)
	X_normalized = preprocessing.normalize(X_scaled, 'l2')

	# Distance Matrix ie sample by sample distances
	distMatrix = pairwise_distances(X_normalized, metric = 'cosine')

	#print 'Fastcluster, cosine distance, average method'
	L = fastcluster.linkage(distMatrix, method = 'average')

	# Dendogram cutting threshold
	dt = 0.5

	indL = sch.fcluster(L, dt*distMatrix.max(), 'distance')
	npindL = np.array(indL)

	# get the first 5 clusters with most number of tweets.
	# (a,b) a occured b # of times or a is also the cluster number.
	freqTwCl = Counter(indL)
	# print "n_clusters", len(freqTwCl)
	# print freqTwCl

	# minimum number of tweets in a cluster = 5
	freq_th = 5
	cluster_score = {}

	# picking the top 5 most populated clusters
	for clfreq in freqTwCl.most_common(5):
		cl = clfreq[0]
		freq = clfreq[1]
		cluster_score[cl] = 0
		if freq >= freq_th:
			clidx = (npindL == cl).nonzero()[0].tolist()
			cluster_centroid = X[clidx].sum(axis=0)
			cluster_tweet = vectorizer.inverse_transform(cluster_centroid)
			for term in np.nditer(cluster_tweet):
				cluster_score[cl] = max(cluster_score[cl], boosted_wdfVoc[str(term).strip()])
			cluster_score[cl] /= freq
			print "cluster_score", cluster_score
			if (debug == 1):
				fout.write("\n")
				fout.write("Score of the below cluster is " + str(cluster_score[cl]))
				fout.write("--------------------------------------------------------------------" + "\n")
				tids = []
				for i in clidx:
					tids.append(map_index_after_cleaning[i])
				for j in tids:
					fout.write(str(tid_to_raw_tweet[j]) + "/n")

	fdata.close()
	fout.close()

