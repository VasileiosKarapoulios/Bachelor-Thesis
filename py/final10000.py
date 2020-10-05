from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import simplejson as json
import datetime
import os
import time
import csv
from pandas import DataFrame
import pandas as pd
#Variables that contains the user credentials to access Twitter API 
access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""
from sklearn.metrics import pairwise_distances_argmin
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn import metrics 
import numpy as np
import nltk
import csv
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import codecs
import time
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import types
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
lemmatizer=WordNetLemmatizer()
stemmer = PorterStemmer() 
from textblob import TextBlob
import random
from random import randrange
from scipy.spatial import distance
from math import sqrt
import math
#This is a basic listener
class StdOutListener(StreamListener):


    def on_data(self, data):
        try:
            global x
            global y
            global NUM_CLUSTERS
            global centroids
            global labels
            global sentences
            global X
            global documents
            global y_kmeans
            global kclusterer
            global Y
            global max
            global documents_score
            global X_sentiment
            global X_sentiment_without_spam
            global Y_sentiment
            global labels_sentiment
            global centroids_sentiment
            global centroids_sentiment_comp
            global centroids_sentiment_comp_without_spam
            global centroids_comp
            global sentences_with_spam
            global labels_with_spam
            global labels_without_spam
            global labels_sentiment_without_spam
            global sentences_without_spam
            global documents_without_spam
            global silhouette_table_for_spam_detection
            global silhouette_table_for_sentiment_trust
            global silhouette_table_for_sentiment_trust_without_spam
            global label_that_defines_spam
            x=x+1
            y=y+1
            data = json.loads(data)
            tweet = data["text"]
            username = data["user"]["name"]
            created = data["created_at"]
            id = data["id"]
            user_id = data["user"]["id"]
            follower = data["user"]["followers_count"]
            friends = data["user"]["friends_count"]
            with open('twitter_data23.csv', mode='a', newline='', encoding="utf8") as twitter_file:
                twitter_file_writer = csv.writer(twitter_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                twitter_file_writer.writerow([created, username, tweet, id, user_id, follower, friends])
            if(x==20):
                with open(r"twitter_data23.csv", encoding = "ISO-8859-1") as f:
                    dis_matrix2 = np.ones(shape=(20,4))
                    
                    reader = csv.reader(f, delimiter=";")
                    documents2=[]
                    documents_score2=[]
                    documents_for_trust2=[]
                    k=0
                    try: 
                        for i in reader:
                            rt_count=0
                            mention_count=0
                            hashtag_count=0
                            onegram_count=0
                            specialcharacters_count=0
                            word_counter=0
                            links_count=0
                            if i[2]:
                                sentence=str(i[2])
                                sentence = sentence.lower()
                                for l in sentence.split():
                                    word_counter=word_counter+1
                                    if(l=='rt'):
                                        rt_count=rt_count+1
                                    if('@' in l):
                                        mention_count=mention_count+1
                                    if('#' in l):
                                        hashtag_count=hashtag_count+1
                                    if re.search(r"([‹‡¨è‰ç°åšæðŸ¥€˜¦ãƒª£†â•–¼Žï¿™àœ¤ïº¾§¥‡³_$%^&*\|}{~])", l):
                                        res=re.search(r"([‹‡¨è‰ç°åšæðŸ¥€˜¦ãƒª£†â•–¼Žï¿™àœ¤ïº¾§¥‡³_$%^&*\|}{~])", l)
                                        specialcharacters_count=specialcharacters_count+len(res.groups())
                                    if re.search('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', l):
                                        res1=re.search('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', l)
                                        links_count=links_count+len(res1.groups())
                                if(mention_count>3):
                                    mention_count=mention_count-3
                                else:
                                    mention_count=0
                                if(hashtag_count>2):
                                    hashtag_count=hashtag_count-2
                                else:
                                    hashtag_count=0
                                if(links_count>1):
                                    links_count=links_count-1
                                else:
                                    links_count=0
                                if(specialcharacters_count>4):
                                    specialcharacters_count=specialcharacters_count-4
                                else:
                                    specialcharacters_count=0
                                score=mention_count*0.25+hashtag_count*0.25+specialcharacters_count*0.1+links_count*0.4
                                dis_matrix2[k,0]=0.25*mention_count
                                dis_matrix2[k,1]=0.25*hashtag_count
                                dis_matrix2[k,2]=0.4*links_count
                                dis_matrix2[k,3]=0.1*specialcharacters_count
                                documents_score2.append(score)
                                sentence=sentence.replace('{html}',"") 
                                cleanr = re.compile('<.*?>')
                                cleantext = re.sub(cleanr, '', sentence)
                                rem_url=re.sub(r'http\S+', '',cleantext)
                                rem_num = re.sub('[0-9]+', '', rem_url)
                                tokenizer = RegexpTokenizer(r'\w+')
                                tokens = tokenizer.tokenize(rem_num)  
                                filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
                                sentence_for_trust2 = filtered_words
                                stem_words=[stemmer.stem(w) for w in filtered_words]
                                lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
                                ff=" ".join(filtered_words)
                                documents2.append([])
                                documents_for_trust2.append([])
                                for j in filtered_words:
                                    documents2[k].append(j)
                                for j in sentence_for_trust2:
                                    documents_for_trust2[k].append(j)
                                k=k+1
                    except BaseException as e:
                        print(e)
                sentences2 = documents2
                Z=[]
                Y2=[]
                print ("========================")
                Z=np.asarray(dis_matrix2)
                kclusterer2 = KMeans(NUM_CLUSTERS, init = 'k-means++', random_state=0)
                kclusterer2.fit(Z)
                Y2=kclusterer2.predict(Z)
                centroids2 = kclusterer2.cluster_centers_
                labels2 = kclusterer2.labels_
                documents_score_for_graph2 = np.asarray(documents_score2)
                documents_score_for_graph2 = documents_score_for_graph2.reshape(-1,1)
                if(y==20):
                    colors=['black','dimgrey','darkgrey', 'lightgrey', 'rosybrown']
                    for j in range(4):
                        ys = documents_score_for_graph2[ Y2==j ]
                        xs = np.random.rand(len(ys))
                        xs=xs.reshape(-1,1)
                        plt.scatter(xs, ys, color=colors[j])
                    for j in range(len(Z)):
                        print ("%s %s" % (labels2[j],  Z[j]))
                    plt.show()
                try:
                    silhouette_score = metrics.silhouette_score(Z, labels2, metric='euclidean')
                    #print ("Silhouette_score: ")
                    #print (silhouette_score)
                except:
                    print(" ")
                labels2_copy=labels2.copy()
                min_centroid = 0
                centroid_counter = 0
                min_centroid_counter = 0
                for i in centroids2:
                    min = 9999
                    for j in centroids_comp:
                        dst = np.sqrt((i[0]-j[0]) ** 2 + (i[1]-j[1]) ** 2 + (i[2]-j[2]) ** 2 + (i[3]-j[3]) ** 2 )#distance.euclidean(i,j)
                        if (dst < min):
                            min = dst
                            min_centroid = min_centroid_counter
                        min_centroid_counter=min_centroid_counter+1
                    for k in range(len(labels2_copy)):
                        if (labels2_copy[k] == centroid_counter):
                            labels2[k] = min_centroid
                            
                            
                    centroid_counter=centroid_counter+1
                    min_centroid_counter=0

                min_centroid = 0
                centroid_counter = 0
                min_centroid_counter = 0
                for i in centroids2:
                    min = 9999
                    for j in centroids_comp:
                        dst = np.sqrt((i[0]-j[0]) ** 2 + (i[1]-j[1]) ** 2 + (i[2]-j[2]) ** 2 + (i[3]-j[3]) ** 2 )#distance.euclidean(i,j)
                        if (dst < min):
                            min = dst
                            min_centroid = min_centroid_counter
                        min_centroid_counter=min_centroid_counter+1
                    length=0
                    for k in range(len(labels2_copy)):
                        if (labels2_copy[k] == centroid_counter):
                            length=length+1
                    centroids_comp[min_centroid,0] = (centroids_comp[min_centroid,0] * len(sentences_with_spam) + centroids2[centroid_counter,0] * length) / (len(sentences_with_spam)+length)
                    centroids_comp[min_centroid,1] = (centroids_comp[min_centroid,1] * len(sentences_with_spam) + centroids2[centroid_counter,1] * length) / (len(sentences_with_spam)+length)
                    centroids_comp[min_centroid,2] = (centroids_comp[min_centroid,2] * len(sentences_with_spam) + centroids2[centroid_counter,2] * length) / (len(sentences_with_spam)+length)
                    centroids_comp[min_centroid,3] = (centroids_comp[min_centroid,3] * len(sentences_with_spam) + centroids2[centroid_counter,3] * length) / (len(sentences_with_spam)+length)
                    centroid_counter=centroid_counter+1
                    min_centroid_counter=0

                labels_with_spam=np.append(labels_with_spam, labels2)
                for j in sentences2:
                    sentences_with_spam.append(j)

#SENTIMENT ANALYSIS
                dis_matrix2_sentiment = np.ones(shape=(len(sentences2),2))
                for i in range(len(sentences2)):
                    analysis = TextBlob(' '.join(sentences2[i]))
                    dis_matrix2_sentiment[i,0]=analysis.sentiment.polarity

                with open('Lexicon_for_trust.csv') as file_csv:
                    csv_reader = csv.reader(file_csv, delimiter=',')
                    counter_trust=0
                    for i in documents_for_trust2:
                        value_of_trust=0
                        for j in range(len(i)):
                            file_csv.seek(0)
                            for row in csv_reader:
                                distance = nltk.edit_distance(i[j], row[0])
                                if(distance<4):
                                    value_of_trust=value_of_trust+0.3
                                    break
                        dis_matrix2_sentiment[counter_trust,1] = value_of_trust
                        counter_trust=counter_trust+1

                for i in range(len(dis_matrix2_sentiment)):
                   dis_matrix2_sentiment[i,1]=2*(dis_matrix2_sentiment[i,1]/max)

                Z_sentiment=[]
                Y2_sentiment=[]
                Z_sentiment=np.asarray(dis_matrix2_sentiment)
                kclusterer2_sentiment=KMeans(NUM_CLUSTERS, init = 'k-means++', random_state=0)
                kclusterer2_sentiment.fit(Z_sentiment)
                centroids2_sentiment=kclusterer2_sentiment.cluster_centers_
                labels2_sentiment=kclusterer2_sentiment.labels_
                colors=['saddlebrown','chocolate','darkred','red','salmon']
                try:
                    silhouette_score = metrics.silhouette_score(Z_sentiment, labels2_sentiment, metric='euclidean')
                    #print ("Silhouette_score: ")
                    #print (silhouette_score)
                except:
                    print(" ")
                ys=[]
                xs=[]
                if(y==20):
                    for j in range(4):
                        for i in range(len(Z_sentiment)):
                            if (labels2_sentiment[i]==j):
                                ys.append(Z_sentiment[i,0])
                                xs.append(Z_sentiment[i,1])
                        plt.scatter(xs, ys, color=colors[j])
                        ys=[]
                        xs=[]
                    for j in range(len(Z_sentiment)):
                        print ("%s %s" % (labels2_sentiment[j],  Z_sentiment[j]))
                    plt.show()

                labels2_sentiment_copy=labels2_sentiment.copy()
                min_centroid = 0
                centroid_counter = 0
                min_centroid_counter = 0
                for i in centroids2_sentiment:
                    min = 9999
                    for j in centroids_sentiment_comp:
                        dst = np.sqrt((i[0]-j[0]) ** 2 + (i[1]-j[1]) ** 2 )#distance.euclidean(i,j)
                        if (dst < min):
                            min = dst
                            min_centroid = min_centroid_counter
                        min_centroid_counter=min_centroid_counter+1
                    for k in range(len(labels2_sentiment_copy)):
                        if (labels2_sentiment_copy[k] == centroid_counter):
                            labels2_sentiment[k] = min_centroid
                    centroid_counter=centroid_counter+1
                    min_centroid_counter=0

                min_centroid = 0
                centroid_counter = 0
                min_centroid_counter = 0
                for i in centroids2_sentiment:
                    min = 9999
                    for j in centroids_sentiment_comp:
                        dst = np.sqrt((i[0]-j[0]) ** 2 + (i[1]-j[1]) ** 2 )#distance.euclidean(i,j)
                        if (dst < min):
                            min = dst
                            min_centroid = min_centroid_counter
                        min_centroid_counter=min_centroid_counter+1

                    length=0
                    for k in range(len(labels2_sentiment_copy)):
                        if (labels2_sentiment_copy[k] == centroid_counter):
                            length=length+1
                    centroids_sentiment_comp[min_centroid,0] = (centroids_sentiment_comp[min_centroid,0] * len(X_sentiment) + centroids2_sentiment[centroid_counter,0] * length) / (len(X_sentiment)+length)
                    centroids_sentiment_comp[min_centroid,1] = (centroids_sentiment_comp[min_centroid,1] * len(X_sentiment) + centroids2_sentiment[centroid_counter,1] * length) / (len(X_sentiment)+length)
                    centroid_counter=centroid_counter+1
                    min_centroid_counter=0

#AFAIRESI SPAM
                cc=0
                for i in range(len(sentences2)):
                    if(labels2[i] == label_that_defines_spam):
                        cc=cc+1
                sentences_to_copy=[]
                sentences_to_copy2=[]
                sentences_to_copy3=np.ones(shape=(len(sentences2)-cc,2))
                labels_to_copy=[]
                counter_for_dis_matrix=0
                for i in range(len(sentences2)):
                    if(labels2[i] != label_that_defines_spam):
                        sentences_to_copy.append(sentences2[i])
                        labels_to_copy = np.append(labels_to_copy, labels2[i])
                        sentences_to_copy2.append(documents_for_trust2[i])
                        sentences_to_copy3[counter_for_dis_matrix,0]=dis_matrix2_sentiment[i,0]
                        sentences_to_copy3[counter_for_dis_matrix,1]=dis_matrix2_sentiment[i,1]
                        counter_for_dis_matrix=counter_for_dis_matrix+1
                labels2_without_spam=labels_to_copy.copy()
                sentences2_without_spam=sentences_to_copy.copy()
                documents_for_trust2_without_spam=sentences_to_copy2.copy()
                dis_matrix2_sentiment_without_spam=sentences_to_copy3.copy()

                Z_sentiment_without_spam=[]
                Y2_sentiment_without_spam=[]
                Z_sentiment_without_spam=np.asarray(dis_matrix2_sentiment_without_spam)
                kclusterer2_sentiment_without_spam=KMeans(NUM_CLUSTERS, init = 'k-means++', random_state=0)
                kclusterer2_sentiment_without_spam.fit(Z_sentiment_without_spam)
                centroids2_sentiment_without_spam=kclusterer2_sentiment_without_spam.cluster_centers_
                labels2_sentiment_without_spam=kclusterer2_sentiment_without_spam.labels_
                colors=['royalblue','midnightblue','mediumblue','slateblue','mediumpurple']
                try:
                    silhouette_score = metrics.silhouette_score(Z_sentiment_without_spam, labels2_sentiment_without_spam, metric='euclidean')
                    #print ("Silhouette_score: ")
                    #print (silhouette_score)
                except:
                    print(" ")
                ys=[]
                xs=[]
                if(y==20):
                    for j in range(4):
                        for i in range(len(Z_sentiment_without_spam)):
                            if (labels2_sentiment_without_spam[i]==j):
                                ys.append(Z_sentiment_without_spam[i,0])
                                xs.append(Z_sentiment_without_spam[i,1])
                        plt.scatter(xs, ys, color=colors[j])
                        ys=[]
                        xs=[]
                    for j in range(len(Z_sentiment_without_spam)):
                        print ("%s %s" % (labels2_sentiment_without_spam[j],  Z_sentiment_without_spam[j]))
                    plt.show()

                labels2_sentiment_copy=labels2_sentiment_without_spam.copy()
                min_centroid = 0
                centroid_counter = 0
                min_centroid_counter = 0
                for i in centroids2_sentiment_without_spam:
                    min = 9999
                    for j in centroids_sentiment_comp_without_spam:
                        dst = np.sqrt((i[0]-j[0]) ** 2 + (i[1]-j[1]) ** 2 )#distance.euclidean(i,j)
                        if (dst < min):
                            min = dst
                            min_centroid = min_centroid_counter
                        min_centroid_counter=min_centroid_counter+1
                    for k in range(len(labels2_sentiment_copy)):
                        if (labels2_sentiment_copy[k] == centroid_counter):
                            labels2_sentiment_without_spam[k] = min_centroid
                    centroid_counter=centroid_counter+1
                    min_centroid_counter=0

                min_centroid = 0
                centroid_counter = 0
                min_centroid_counter = 0
                for i in centroids2_sentiment_without_spam:
                    min = 9999
                    for j in centroids_sentiment_comp_without_spam:
                        dst = np.sqrt((i[0]-j[0]) ** 2 + (i[1]-j[1]) ** 2 )#distance.euclidean(i,j)
                        if (dst < min):
                            min = dst
                            min_centroid = min_centroid_counter
                        min_centroid_counter=min_centroid_counter+1
                    length=0
                    for k in range(len(labels2_sentiment_copy)):
                        if (labels2_sentiment_copy[k] == centroid_counter):
                            length=length+1
                    centroids_sentiment_comp_without_spam[min_centroid,0] = (centroids_sentiment_comp_without_spam[min_centroid,0] * len(X_sentiment_without_spam) + centroids2_sentiment_without_spam[centroid_counter,0] * length) / (len(X_sentiment_without_spam)+length)
                    centroids_sentiment_comp_without_spam[min_centroid,1] = (centroids_sentiment_comp_without_spam[min_centroid,1] * len(X_sentiment_without_spam) + centroids2_sentiment_without_spam[centroid_counter,1] * length) / (len(X_sentiment_without_spam)+length)
                    centroid_counter=centroid_counter+1
                    min_centroid_counter=0

                X_sentiment_without_spam=np.array(X_sentiment_without_spam)
                Z_sentiment_without_spam=np.array(Z_sentiment_without_spam)
                X_sentiment_without_spam=np.concatenate((X_sentiment_without_spam,Z_sentiment_without_spam))
                labels_sentiment_without_spam=np.append(labels_sentiment_without_spam,labels2_sentiment_without_spam)
                X=np.array(X)
                Z=np.array(Z)
                X=np.concatenate((X,Z))
                X_sentiment=np.array(X_sentiment)
                Z_sentiment=np.array(Z_sentiment)
                X_sentiment=np.concatenate((X_sentiment,Z_sentiment))
                documents_score.extend(documents_score2)
                Y=np.array(Y)
                Y2=np.array(Y2)
                Y=np.concatenate((Y,Y2))
                Y_sentiment=np.array(Y_sentiment)
                Y2_sentiment=np.array(Y2_sentiment)
                Y_sentiment=np.concatenate((Y_sentiment,Y2_sentiment))
                for j in sentences2:
                    sentences.append(j)
                labels=np.append(labels, labels2)
                labels_sentiment=np.append(labels_sentiment, labels2_sentiment)

#SINOLO DOCUMENTS SCORE
                ys=[]
                xs=[]
                colors=['black','dimgrey','darkgrey', 'lightgrey', 'rosybrown']
                documents_score_for_graph2 = np.asarray(documents_score)
                documents_score_for_graph2 = documents_score_for_graph2.reshape(-1,1)
                silhouette_score = metrics.silhouette_score(X, labels_with_spam, metric='euclidean')
                print ("Silhouette_score: ")
                print (silhouette_score)
                silhouette_table_for_spam_detection.append(silhouette_score)
                if((y==20) or (y==500) or (y==1000) or (y==1500) or (y==2000)):
                    for j in range(4):
                        ys = documents_score_for_graph2[ labels_with_spam==j ]
                        xs = np.random.rand(len(ys))
                        xs=xs.reshape(-1,1)
                        plt.scatter(xs, ys, color=colors[j])
                    plt.show()


#SINOLO SENTIMENT
                colors=['saddlebrown','chocolate','darkred','red','salmon']
                ys=[]
                xs=[]
                silhouette_score = metrics.silhouette_score(X_sentiment, labels_sentiment, metric='euclidean')
                print ("Silhouette_score: ")
                print (silhouette_score)
                silhouette_table_for_sentiment_trust.append(silhouette_score)
                if((y==20) or (y==500) or (y==1000) or (y==1500) or (y==2000)):
                    for j in range(4):
                        for i in range(len(X_sentiment)):
                            if (labels_sentiment[i]==j):
                                ys.append(X_sentiment[i,0])
                                xs.append(X_sentiment[i,1])
                        plt.scatter(xs, ys, color=colors[j])
                        ys=[]
                        xs=[]
                    plt.show()

#SINOLO SENTIMENT XWRIS SPAM
                colors=['royalblue','midnightblue','mediumblue','slateblue','mediumpurple']
                ys=[]
                xs=[]
                silhouette_score = metrics.silhouette_score(X_sentiment_without_spam, labels_sentiment_without_spam, metric='euclidean')
                print ("Silhouette_score: ")
                print (silhouette_score)
                silhouette_table_for_sentiment_trust_without_spam.append(silhouette_score)
                if((y==20) or (y==500) or (y==1000) or (y==1500) or (y==2000)):
                    for j in range(4):
                        for i in range(len(X_sentiment_without_spam)):
                            if (labels_sentiment_without_spam[i]==j):
                                ys.append(X_sentiment_without_spam[i,0])
                                xs.append(X_sentiment_without_spam[i,1])
                        plt.scatter(xs, ys, color=colors[j])
                        ys=[]
                        xs=[]
                    plt.show()


                '''print(silhouette_table_for_spam_detection)
                print(silhouette_table_for_sentiment_trust)
                print(silhouette_table_for_sentiment_trust_without_spam)
                print("Sentiment/Users")
                print(sentiment_users)
                print("Sentiment w/o spam/Users")
                print(sentiment_users_no_spam)
                print("Trust/Users")
                print(trust_users)
                print("Trust w/o spam/Users")
                print(trust_users_no_spam)'''

                os.remove("twitter_data23.csv")
                x=0
                if(y==2000):
                    sys.exit()
            return True
        except BaseException as e:
            print('Failed on data')
            print(e)
    def on_error(self, status):
        print (status)


if __name__ == '__main__':
    if(os.path.isfile("twitter_data23.csv")):
        os.remove("twitter_data23.csv")
    x=0
    y=0
    with open(r"twitter_data_final.csv", encoding = "ISO-8859-1") as f:
        reader = csv.reader(f, delimiter=";")
        documents=[]
        documents_for_trust=[]
        documents_score=[]
        k=0
        dis_matrix = np.ones(shape=(10000,4))
        try:
            for counterlines in range(10000):
                for i in reader:
                    rt_count=0
                    mention_count=0
                    hashtag_count=0
                    onegram_count=0
                    specialcharacters_count=0
                    word_counter=0
                    links_count=0
                    if i[2]:
                        sentence=str(i[2])
                        sentence = sentence.lower()
                        for l in sentence.split():
                            word_counter=word_counter+1
                            if(l=='rt'):
                                rt_count=rt_count+1
                            if('@' in l):
                                mention_count=mention_count+1
                            if('#' in l):
                                hashtag_count=hashtag_count+1
                            if re.search(r"([‹‡¨è‰ç°åšæðŸ¥€˜¦ãƒª£†â•–¼Žï¿™àœ¤ïº¾§¥‡³_$%^&*\|}{~])", l):
                                res=re.search(r"([‹‡¨è‰ç°åšæðŸ¥€˜¦ãƒª£†â•–¼Žï¿™àœ¤ïº¾§¥‡³_$%^&*\|}{~])", l)
                                specialcharacters_count=specialcharacters_count+len(res.groups())
                            if re.search('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', l):
                                res1=re.search('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', l)
                                links_count=links_count+len(res1.groups())
                        if(mention_count>3):
                            mention_count=mention_count-3
                        else:
                            mention_count=0
                        if(hashtag_count>2):
                            hashtag_count=hashtag_count-2
                        else:
                            hashtag_count=0
                        if(links_count>1):
                            links_count=links_count-1
                        else:
                            links_count=0
                        if(specialcharacters_count>4):
                            specialcharacters_count=specialcharacters_count-3
                        else:
                            specialcharacters_count=0
                        score=mention_count*0.25+hashtag_count*0.25+specialcharacters_count*0.1+links_count*0.4
                        dis_matrix[k,0]=0.25*mention_count
                        dis_matrix[k,1]=0.25*hashtag_count
                        dis_matrix[k,2]=0.4*links_count
                        dis_matrix[k,3]=0.1*specialcharacters_count
                        documents_score.append(score)
                        sentence=sentence.replace('{html}',"")
                        cleanr = re.compile('<.*?>')
                        cleantext = re.sub(cleanr, '', sentence)
                        rem_url=re.sub(r'http\S+', '',cleantext)
                        rem_num = re.sub('[0-9]+', '', rem_url)
                        tokenizer = RegexpTokenizer(r'\w+')
                        tokens = tokenizer.tokenize(rem_num)
                        filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
                        sentence_for_trust = filtered_words
                        stem_words=[stemmer.stem(w) for w in filtered_words]
                        lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
                        ff=" ".join(filtered_words)
                        documents.append([])
                        documents_for_trust.append([])
                        for j in filtered_words:
                            documents[k].append(j)
                        for j in sentence_for_trust:
                            documents_for_trust[k].append(j)
                        k=k+1
        except BaseException as e:
            print(e)
    sentences = documents
    silhouette_table_for_spam_detection = []
    silhouette_table_for_sentiment_trust = []
    silhouette_table_for_sentiment_trust_without_spam = []
    X=[]
    Y=[]
    print ("========================")
    X=np.asarray(dis_matrix)
    NUM_CLUSTERS=4
    kclusterer = KMeans(NUM_CLUSTERS, init='k-means++', random_state=0)
    kclusterer.fit(X)
    Y=kclusterer.predict(X)
    centroids = kclusterer.cluster_centers_
    centroids_comp = centroids
    labels = kclusterer.labels_
    print ("Cluster id labels for inputted data")
    print (labels)
    print ("Centroids data")
    print (centroids)
    silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
    print ("Silhouette_score: ")
    print (silhouette_score)
    silhouette_table_for_spam_detection.append(silhouette_score)
    documents_score_for_graph = np.asarray(documents_score)
    documents_score_for_graph=documents_score_for_graph.reshape(-1,1)
    colors=['black','dimgrey','darkgrey', 'lightgrey', 'rosybrown']
    for j in range(4):
        ys = documents_score_for_graph[ Y==j ]
        xs=np.random.rand(len(ys))
        xs=xs.reshape(-1,1)
        plt.scatter(xs, ys, color=colors[j])
    for j in range(len(X)):
        print ("%s %s" % (labels[j],  X[j]))
    plt.show()

    gg=0
    cg=0
    cc=0
    med=0
    for i in range(len(sentences)):
        if(labels[i] == 0):
            cg=cg+1
        if(labels[i] == 1):
            gg=gg+1
        if(labels[i] == 2):
            med=med+1
        if(labels[i] == 3):
            cc=cc+1
    stath=(cc/len(sentences))*100
    stathgg=(gg/len(sentences))*100
    stathmed=(med/len(sentences))*100
    stathcg=(cg/len(sentences))*100
    print("Percent of black tweets ")
    print(stathcg, "%")
    print(cg, "from 10000")
    print("Percent of dimgrey tweets ")
    print(stathgg, "%")
    print(gg, "from 10000")
    print("Percent of lightgrey tweets ")
    print(stath,"%")
    print(cc, "from 10000")
    print("Percent of darkgrey tweets ")
    print(stathmed,"%")
    print(med, "from 10000")

    max_tweet_score=0
    label_that_defines_spam=0
    for i in range(len(X)):
        stable=0
        for j in X[i]:
            stable=stable+j
        if (stable>max_tweet_score):
            max_tweet_score=stable
            label_that_defines_spam=labels[i].copy()
    print("Cluser that defines spam")
    print(label_that_defines_spam)

    dis_matrix_sentiment = np.ones(shape=(len(sentences),2))
    for i in range(len(sentences)):
        analysis = TextBlob(' '.join(sentences[i]))
        dis_matrix_sentiment[i,0] = analysis.sentiment.polarity

    start = time.time()
    print("Starting...")

    with open('Lexicon_for_trust.csv') as file_csv:
        csv_reader = csv.reader(file_csv, delimiter=',')
        counter_trust=0
        for i in documents_for_trust:
            value_of_trust=0
            for j in range(len(i)):
                file_csv.seek(0)
                for row in csv_reader:
                    distance = nltk.edit_distance(i[j], row[0])
                    if(distance<4):
                        value_of_trust=value_of_trust+0.3
                        break
                    
            dis_matrix_sentiment[counter_trust,1] = value_of_trust
            counter_trust=counter_trust+1
    end = time.time()
    print(end-start)

    max=0
    for i in range(len(dis_matrix_sentiment)):
        if(dis_matrix_sentiment[i,1]>max):
           max=dis_matrix_sentiment[i,1]

    for i in range(len(dis_matrix_sentiment)):
       dis_matrix_sentiment[i,1]=2*(dis_matrix_sentiment[i,1]/max)

    X_sentiment=[]
    Y_sentiment=[]
    X_sentiment=np.asarray(dis_matrix_sentiment)
    kclusterer_sentiment = KMeans(NUM_CLUSTERS, init='k-means++', random_state=0)
    kclusterer_sentiment.fit(X_sentiment)
    Y_sentiment=kclusterer_sentiment.predict(X_sentiment)
    centroids_sentiment = kclusterer_sentiment.cluster_centers_
    centroids_sentiment_comp = centroids_sentiment
    labels_sentiment = kclusterer_sentiment.labels_
    print ("Cluster id labels for sentiment analysis data")
    print (labels_sentiment)
    print ("Centroids for sentiment data")
    print (centroids_sentiment)
    silhouette_score = metrics.silhouette_score(X_sentiment, labels_sentiment, metric='euclidean')
    print ("Silhouette_score for sentiment: ")
    print (silhouette_score)
    silhouette_table_for_sentiment_trust.append(silhouette_score)
    ys=[]
    xs=[]
    colors=['saddlebrown','chocolate','darkred','red','salmon']
    for j in range(4):
        for i in range(len(X_sentiment)):
            if (labels_sentiment[i]==j):
                ys.append(X_sentiment[i,0])
                xs.append(X_sentiment[i,1])
        plt.scatter(xs, ys, color=colors[j])
        ys=[]
        xs=[]
    gt=0
    nt=0
    bt=0
    bb=0
    for i in range(len(sentences)):
        if(labels_sentiment[i] == 3):
            bb=bb+1
        if(labels_sentiment[i] == 2):
            gt=gt+1
        if(labels_sentiment[i] == 1):
            nt=nt+1
        if(labels_sentiment[i] == 0):
            bt=bt+1
    stathgt=(gt/len(sentences))*100
    stathnt=(nt/len(sentences))*100
    stathbt=(bt/len(sentences))*100
    stathbb=(bb/len(sentences))*100
    print("Percent of red tweets ")
    print(stathbb,"%")
    print("Percent of darkred tweets ")
    print(stathgt,"%")
    print("Percent of chocolate tweets ")
    print(stathnt,"%")
    print("Percent of saddlebrown tweets ")
    print(stathbt,"%")
    for j in range(len(X_sentiment)):
        print ("%s %s" % (labels_sentiment[j],  X_sentiment[j]))
    plt.show()

    sentences_with_spam = sentences.copy()
    labels_with_spam = labels.copy()
    sentences_to_copy=[]
    sentences_to_copy2=[]
    labels_to_copy=[]
    cc=0
    for i in range(len(sentences)):
        if(labels[i] == label_that_defines_spam):
            cc=cc+1
    print("Number of spams")
    print(cc)
    stath=(cc/len(sentences))*100
    print("Percent of spam tweets ")
    print(stath,"%")
    counter_for_dis_matrix=0
    sentences_to_copy3=np.ones(shape=(len(sentences)-cc,2))
    for i in range(len(sentences)):
        if(labels[i] != label_that_defines_spam):
            sentences_to_copy.append(sentences[i])
            labels_to_copy = np.append(labels_to_copy, labels[i])
            sentences_to_copy2.append(documents_for_trust[i])
            sentences_to_copy3[counter_for_dis_matrix,0]=dis_matrix_sentiment[i,0]
            sentences_to_copy3[counter_for_dis_matrix,1]=dis_matrix_sentiment[i,1]
            counter_for_dis_matrix=counter_for_dis_matrix+1

    labels_without_spam=labels_to_copy.copy()
    sentences_without_spam=sentences_to_copy.copy()
    documents_for_trust_without_spam=sentences_to_copy2.copy()
    dis_matrix_sentiment_without_spam=sentences_to_copy3.copy()

    X_sentiment_without_spam=[]
    Y_sentiment_without_spam=[]
    X_sentiment_without_spam=np.asarray(dis_matrix_sentiment_without_spam)
    kclusterer_sentiment_without_spam = KMeans(NUM_CLUSTERS, init='k-means++', random_state=0)
    kclusterer_sentiment_without_spam.fit(X_sentiment_without_spam)
    Y_sentiment_with_spam=kclusterer_sentiment_without_spam.predict(X_sentiment_without_spam)
    centroids_sentiment_without_spam = kclusterer_sentiment_without_spam.cluster_centers_
    centroids_sentiment_comp_without_spam = centroids_sentiment_without_spam
    labels_sentiment_without_spam = kclusterer_sentiment_without_spam.labels_
    print ("Cluster id labels for sentiment analysis data")
    print (labels_sentiment_without_spam)
    print ("Centroids for sentiment data")
    print (centroids_sentiment_without_spam)
    silhouette_score = metrics.silhouette_score(X_sentiment_without_spam, labels_sentiment_without_spam, metric='euclidean')
    print ("Silhouette_score for sentiment: ")
    print (silhouette_score)
    silhouette_table_for_sentiment_trust_without_spam.append(silhouette_score)
    ys=[]
    xs=[]
    colors=['royalblue','midnightblue','mediumblue','slateblue','mediumpurple']
    for j in range(4):
        for i in range(len(X_sentiment_without_spam)):
            if (labels_sentiment_without_spam[i]==j):
                ys.append(X_sentiment_without_spam[i,0])
                xs.append(X_sentiment_without_spam[i,1])
        plt.scatter(xs, ys, color=colors[j])
        ys=[]
        xs=[]
    gt=0
    nt=0
    bt=0
    bb=0
    for i in range(len(sentences_without_spam)):
        if(labels_sentiment_without_spam[i] == 3):
            bb=bb+1
        if(labels_sentiment_without_spam[i] == 2):
            gt=gt+1
        if(labels_sentiment_without_spam[i] == 1):
            nt=nt+1
        if(labels_sentiment_without_spam[i] == 0):
            bt=bt+1
    stathgt=(gt/len(sentences_without_spam))*100
    stathnt=(nt/len(sentences_without_spam))*100
    stathbt=(bt/len(sentences_without_spam))*100
    stathbb=(bb/len(sentences_without_spam))*100
    print("Percent of slateblue tweets ")
    print(stathbb,"%")
    print("Percent of mediumblue tweets ")
    print(stathgt,"%")
    print("Percent of midnightblue tweets ")
    print(stathnt,"%")
    print("Percent of royalblue tweets ")
    print(stathbt,"%")
    for j in range(len(X_sentiment_without_spam)):
        print ("%s %s" % (labels_sentiment_without_spam[j],  X_sentiment_without_spam[j]))
    plt.show()
    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filters Twitter Streams to capture data by the keywords: Starbucks
    stream.filter(languages=["en"],track=['Starbucks'])
