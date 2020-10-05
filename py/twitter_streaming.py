#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import simplejson as json
import datetime
import os
import time
import csv
import sys
from pandas import DataFrame
import pandas as pd
#Variables that contains the user credentials to access Twitter API 
access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    #with open('tweet.json', 'a', encoding='utf8') as file:
        #json.dumps(status._json, file,sort_keys = True,indent = 4)

    
    def on_data(self, data):
        try:
            global x
            x=x+1
            data = json.loads(data)
            tweet = data["text"]
            username = data["user"]["name"]
            created = data["created_at"]
            id = data["id"]
            user_id = data["user"]["id"]
            follower = data["user"]["followers_count"]
            friends = data["user"]["friends_count"]
            with open('twitter_data_final.csv', mode='a', newline='', encoding="utf8") as twitter_file:
                twitter_file_writer = csv.writer(twitter_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                twitter_file_writer.writerow([created, username, tweet, id, user_id, follower, friends])
            print("Number of tweets downloaded:")
            print(x)
            return True
        except BaseException as e:
            print('Failed on data')
            print(e)
    def on_error(self, status):
        print (status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    x=0
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby' 
    stream.filter(languages=["en"],track=['Starbucks'])
