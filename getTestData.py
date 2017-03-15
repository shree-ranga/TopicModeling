# Event Detection from Tweets
# -*- coding: utf-8 -*- 

# Run this code first.

__author__ = 'Shree Ranga Raju'

import tweepy
import pymongo
import pprint
import json

def init_twitter_API():
	# Twitter API Keys and Secrets
	consumer_key = 'xBtsxLlR91JCc3oQy2yEY89Vk'
	consumer_secret = 'qxnhmkqIPEttz8N3mQIlU471cE3amI2e0yOqLvQJnslAdpRh5U'
	access_token = '144670619-QZ65Sa4E0189XA1dCks8ZSte92mr5v2vfp0cDbea'
	access_secret = '6ZZ6Uj1omxUi3Tpn5e9U57dVJ1weZ2AF7KTHNIDj3zERe'

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_secret)

	api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)
	return api

def init_mongodb():
	# Mongodb instance on port 27017 (Default monogdb port).
	client = pymongo.MongoClient('localhost:27017')
	db = client.NLPTestData
	return db

def parse_json_data(data):
	tweet = json.loads(data)
	date = tweet['created_at']
	tweet_id = tweet['id']
	screen_name = tweet['user']['name']
	# Consider retweet as a tweet from the user
	if 'retweeted_status' in tweet:
		text = tweet['retweeted_status']['text']
	else:
		text = tweet['text']

	return [date, tweet_id, screen_name, text]

def push_tweet_to_db(date, tweet_id, screen_name, text):
	db_json = {}
	db_json['tweet_id'] = tweet_id
	db_json['screen_name'] = screen_name
	db_json['tweet_text'] = text
	db_json['date'] = date
	#print db_json
	# DB name is MyTweetsDB and collection name is tweets
	insert_json_db = db.tweets.insert_one(json.loads(json.dumps(db_json)))
	

def get_data(user_name):
	# Gets last 100 tweets from each user from the moment you execute this code.
	# For instance, if I run this code at 10am Jan 10th 2016 I'd be receiving 
	# all the 100 tweets that were published before that time.
	for tweets in tweepy.Cursor(api.search, q = user_name, languages = 'en').items(5):
		data = json.dumps(tweets._json)
		[date, tweet_id, screen_name, text] = parse_json_data(data)
		push_tweet_to_db(date, tweet_id, screen_name, text)

if __name__ == '__main__':

	api = init_twitter_API()
	print 'Initialized Twitter API.' + '\n'

	db = init_mongodb()
	print 'Initialized Mongodb Instance.' + '\n' 

	user_names = ['oscars', 'superbowl', 'grammy', 'trump', 'election2016']

	for user_name in user_names:
		get_data(user_name)

	print 'Data is stored in db. It is ready for processing.'











