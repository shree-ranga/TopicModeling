# Event Detection from Tweets
# -*- coding: utf-8 -*-

# Run this code first. This file gets the tweet data from 10 different news companies.

__author__ = 'Shree Ranga Raju'

import tweepy
import pprint
import codecs
import json

def init_twitter_API():
	# Twitter API Keys and Secrets
	consumer_key = 'pWA5z44PjdxpkT6bcX9vrrjdQ'
	consumer_secret = 's347W3diJm0zoSYvKDMBXi7tHKlaZyKkqovPBImfWsIC2ayc8o'
	access_token = '144670619-MWz10ABNcnBiMei1ljFRZ1Fj9dVblOn8ZPPUNvZj'
	access_secret = 'Vvk3WF9MN6GgOnyA9Q1qXTM37v740U4KPi59wx5CNUSGt'

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_secret)

	api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)
	return api


def parse_json_data(data):
	tweet = json.loads(data)
	# date = tweet['created_at']
	# tweet_id = tweet['id']
	# screen_name = tweet['user']['name']
	# Consider retweet as a tweet from the user
	if 'retweeted_status' in tweet:
		text = tweet['retweeted_status']['text']
	else:
		text = tweet['text']
	# return [date, tweet_id, screen_name, text]
	return text

# def push_tweet_to_file(text):
# 	# db_json = {}
# 	# db_json['tweet_id'] = tweet_id
# 	# db_json['screen_name'] = screen_name
# 	# db_json['tweet_text'] = text
# 	# db_json['date'] = date
# 	# DB name is MyTweetsDB and collection name is tweets

# 	print "wrote to file"

def get_data(user_name, fdata):
	# Gets last 100 tweets from each user from the moment you execute this code.
	# For instance, if I run this code at 10am Jan 10th 2016 I'd be receiving
	# all the 100 tweets that were published before that time.
	for tweets in tweepy.Cursor(api.user_timeline, id = user_name).items(3):
		data = json.dumps(tweets._json)
		tweet_text = parse_json_data(data)
		# push_tweet_to_file(tweet_text)
		print tweet_text
		fdata.write(str(tweet_text.encode('utf8')) + "\n")


if __name__ == '__main__':

	api = init_twitter_API()
	print 'Initialized Twitter API.' + '\n'

	# user_names = ['nytimes', 'cnn', 'abc', 'ajenglish', 'bbcnews', 'washingtonpost', 'usatoday', 'thetimes', 'cnet', 'telegraph']

	user_names = ['nytimes']

	fdata = fdata = open("data.txt", "w")

	for user_name in user_names:
		get_data(user_name, fdata)

	fdata.close()











