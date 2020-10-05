# Bachelor-Thesis
Remove unwanted messages to more effectively study consumers’ behavior on Twitter

**Description** 

K-Means clustering on live-streaming data (using Twitter’s streaming API) to detect spam/non-spam tweets – “Starbucks” Case Study (#starbucks) – considering 4 features in each post and apply sentiment analysis and trust analysis using NLTK library and NRC Emotion Lexicon.

**Goal**

The goal is to detect and remove spam tweets, in order to study better the behavior of the consumers, for example how they feel about the company or if they trust them. 

**Process** 

  First of all, the data were collected using Twitter's Streaming API which provides read-only permissions to the data. The data are stored in a twitter_data.csv file and more specifically, date and time - user's name - user's id - tweet - tweet's id - number of followers - number of followings - are stored. At the beginning 10.000 tweets are collected. Due to the nature of the tweets, a pre-processing of the data is needed, thus tokenization, stemming/lemmatization, upper to lower case methods are applied and moreover html/https links, numbers and stop-words are removed. Due to the encoding of the tweets, emojis and characters that origin from non-english keyboard are stored with ASCII codes. The reason for not removing them, is that tweets that contain massive amounts of those, refer usually to spams. Last but not least, number of mentions (@), number of hashtags (#), links and words that contain ASCII codes are counted. These will represent our 4 features per each tweet for the clustering algorithm, as we assume that tweets that contain certain amounts of those features refer to spam posts. Below is an example of this preprocessing mentioned before (the username is erased manually on purpose):
  
![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/Preprocessing.PNG?raw=true)

  In the next step, there a n x 4 matrix created, where n is the number of tweets and 4 the number of features (the previously counted ones). The weights are chosen after experiments and are 0.25,0.25,0.4,0.1 respectively. In addition to this, mentions and hashtags are taken into consideration if and only if they are counted more than 3 and 2 respectively, whereas links if they are more than 1 and ASCII code words more than 4. The value in each cell of the matrix is a result of the product between the weight of each feature and the corresponding difference of counted number-threshold of the feature. I.e if there are 5 mentions in a tweet, the value of the corresponding cell will be 
(5-3) * 0.25. This is a way to deal with outliers. The next picture shows a sketch of the matrix:

![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/Matrix.PNG?raw=true)

  After this is done, we apply the clustering K-Means algorithm for 4 cluster. It's worth mentioning that the sklearn library is used. The algorithm returns the following clusters: 
  
![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/Clustering10000.PNG?raw=true)
![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/PercentTweets.PNG?raw=true)

The y axis in the left graph corresponds to the sum of all features and the x axis is a random number. These are used **only** to spread and illustrate the clusters. The cluster that defines spam is the cluster with the color dimgrey and they apart ~0.41% of the collected tweets. 

  To evaluate the results, we use the Silhouette score, which uses the Euclidean distance. It basically shows how close each observation is compared to other observations of the same cluser and how far an observation is compared to observations in different clusters. The Silhouette score in this clustering was calculated as: ~0.98.
  
   In the next step, we perform sentiment analysis and trust factor calculation:
   1)Sentiment analysis is an analysis that shows the sentiment of the user, calculates a score which varies from -1 to +1 and \[-1,0) means negative sentiment, 0 means neutral sentiment and (0,1] means positive sentiment. Edit_distance from library NLTK is used and is a way to compare if two words are equal. It contains Levenshtein distance functionalities, which refers to removal, addition or replacement of a character in the string in order to convert one word to the other. For example, the Levenshtein distance between "kitten" and "sitting" is 3, because the actions needed to convert the first word to the second are:
      kitten → sitten (replacement of "s" with "k")
      sitten → sittin (replacement of "i" with "e")
      sittin → sitting (addition of "g" in the end)
      
   The threshold used in this case, so that a word is similar to another one is 3 and the lexicon that we used is NRC Emotion Lexicon, a little bit editied to follow the needs of this project. 
   2) The trust factor shows how much a customer trusts the company and the Lexicon_for_trust is used for this purpose. It contains words that declare trust.  
   
Thus, a matrix is created that contains for each tweet the sentiment polarity and the trust value.
   
![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/SentimentTrust.PNG?raw=true)  

Having created this table, we apply K-Means again on this n x 2 matrix for 4 clusters. 

![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/SentimentTrust10000.PNG?raw=true)
![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/PercentSentimentTrust.PNG?raw=true) 

The x axis corresponds to the trust-value and the y axis corresponds to the sentiment polarity. We can observe, that there are clusters created with customers with very little trust to the company (chocolate), customers with neutral values for trust and positive (darkred) or negative(red) sentiment polarity and finally customers with high values for trust and positive sentiment polarity (saddlebrown). Again, the Silhouette score is used to evaluate the clustering and it turns out to be ~0.5003. 

  Now we remove the spam tweets and we perform again the same analysis.
  
![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/SentimentTrust10000NoSpam.PNG?raw=true)
![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/PercentSentimentTrust10000NoSpam.PNG?raw=true) 

There are clusters created with customers with very little trust to the company (midnightblue), customers with neutral values for trust and positive (royalblue) or negative(mediumblue) sentiment polarity and finally customers with high values for trust and positive sentiment polarity (slateblue). We observe now that we have removed the spam messages, the percentages have changed with the only significant change refering to the customers with neutral values for trust and positive values of sentiment polarity from ~11.2 to 11.68. These kind of customers, seem to like the company, thus it was good to remove those spam messages so that they are more distinct. Again, the Silhouette score is ~0.5008.

  At this point, we proceed to the live-streaming part, in which tweets keep coming and the previous process is repeated over and over again. In order to achieve this, the system processes the data every 20 new tweets. So, every 20 new tweets all of the above steps are repeated with the only difference that now we include the previous clustering along with the new 20 tweets. This means that, after clustering the new 20 data, the new cluster centers and the old cluster centers are compared and the new data are assigned to the cluster that their center is closest to regarding the old cluster centers. In the next graphs the process for 20 new tweets is illustrated:

The 20 new tweets clustering

![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/20newtweetsClustering.PNG?raw=true)

The 20 new tweets sentiment analysis-trust factor

![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/20newtweetsSentimentTrust.PNG?raw=true) 

The 20 new tweets sentiment analysis-trust factor after removing spam

![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/20newtweetsSentimentTrustNoSpam.PNG?raw=true)

The connection of the 20 new tweets with the previous clustering

![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/ClusterConnection1020.PNG?raw=true)

The connection of the 20 new tweets sentiment analysis-trust factor with the previous collection

![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/SentimentTrustConnection1020.PNG?raw=true)

The connection of the 20 new tweets sentiment analysis-trust factor with the previous collection after removing spam

![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/SentimentTrustConnection1020NoSpam.PNG?raw=true) 


Below we present the graphs after 2000 new tweets 

![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/SentimentTrust12000.PNG?raw=true) 

![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/SentimentTrust12000NoSpam.PNG?raw=true) 

Below, the Silhouette scores are presented in the barplot

![alt text](https://github.com/VasileiosKarapoulios/Bachelor-Thesis/blob/main/Screenshots/Silhouette.PNG?raw=true)   

As we observe, as we keep adding new observations to the clustering, the quality gets worse, which is reasonable.

  All in all, the removal of the spam messages improve the study of the consumer behavior, as the sentiment and the trust factor and they correspond to the reality. The clustering as we can conclude from the silhouette factor has a slightly better quality after removing spam tweets. We also observe from the Silhouette graph that the less the quality is in the initial clustering (black bar) the higher the difference is in the quality of the clustering for sentiment analysis when including/excluding spam messages. Thus, the removal of the spam messages doesn't affect negatively the study, it even improves it slightly.  

**Improvement** 

As we observed, the quality of the clustering gets worse and worse as we add more observations to the existing clusters. Thus, a good improvement would be to re-run the K-Means clustering from the beginning after a specific number of iterations, for example after every 2000 tweets. 
