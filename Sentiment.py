#%%
from Preprocessing import filtered_tweets_biden, filtered_tweets_trump, filtered_tweets
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#%%
## Adding additional columns to store sentiment analysis results
filtered_tweets["positive"] = 0.0
filtered_tweets["neutral"] = 0.0
filtered_tweets["negative"] = 0.0
filtered_tweets["compound"] = 0.0
filtered_tweets["likely_party"] = 0  # 0 for unsure, 1 for dem, 2 for rep

#%%
from process_tweet import process_tweet
sid = SentimentIntensityAnalyzer()

# Process tweets and calculate sentiment scores
def analyze_sentiment(row):
    tweet = process_tweet(row["tweet"])
    sentiment_dict = sid.polarity_scores(tweet)
    row['positive'] = sentiment_dict['pos']
    row['neutral'] = sentiment_dict['neu']
    row['negative'] = sentiment_dict['neg']
    row['compound'] = sentiment_dict['compound']
    
    if (sentiment_dict['compound'] < 0.05) and (sentiment_dict['compound'] >= 0) :
        row['likely_party'] = 0
    elif row["tweet_about"] == "biden":
        row['likely_party'] = 1 if sentiment_dict['compound'] > 0 else 2
    elif row["tweet_about"] == "trump":
        row['likely_party'] = 2 if sentiment_dict['compound'] > 0 else 1
    
    return row

# Apply the sentiment analysis function to each row
filtered_tweets = filtered_tweets.apply(analyze_sentiment, axis=1)

#%%
import pickle as pkl
with open('filtered_tweets.pkl', 'wb') as f:
    pkl.dump(filtered_tweets, f)

# %%
import pickle as pkl
loaded = pkl.load(open('filtered_tweets.pkl', 'rb'))
print(loaded.columns)
nue,dem,rep = 0,0,0
for i in loaded["likely_party"]:
    if i == 0:
        nue += 1
    elif i == 1:
        dem += 1
    else:
        rep += 1
print(nue, dem, rep)
print(nue/len(loaded), dem/len(loaded), rep/len(loaded))
print(len(loaded[loaded["compound"] == 0])/len(loaded))
# %%
#make a file with 100 random nutral tweets
import random
nue_tweets = loaded[loaded["likely_party"] == 0]
nue_tweets = nue_tweets.sample(n=100)
nue_tweets[["tweet", "compound"]].to_csv("nue_tweets.csv", index=False)


positive_tweets = loaded[loaded["compound"] > 0.05]
negative_tweets = loaded[loaded["compound"] < 0]

positive_tweets = positive_tweets.sample(n=100)
negative_tweets = negative_tweets.sample(n=100)

positive_tweets[["tweet", "compound","likely_party"]].to_csv("positive_tweets.csv", index=False)
negative_tweets[["tweet", "compound","likely_party"]].to_csv("negative_tweets.csv", index=False)

# %%
testtweet ="lock him up"
a = sid.polarity_scores(testtweet)
print(a)
# %%
# # Process tweets and calculate sentiment scores using TextBlob
# from textblob import TextBlob
# def analyze_sentiment_textblob(row):
#     tweet = process_tweet(row["tweet"])
#     analysis = TextBlob(tweet)
#     row['polarity'] = analysis.sentiment.polarity
#     row['subjectivity'] = analysis.sentiment.subjectivity
    
#     if abs(analysis.sentiment.polarity) < 0.1:
#         row['likely_party'] = 0
#     elif row["tweet_about"] == "biden":
#         row['likely_party'] = 1 if analysis.sentiment.polarity > 0 else 2
#     elif row["tweet_about"] == "trump":
#         row['likely_party'] = 2 if analysis.sentiment.polarity > 0 else 1
    
#     return row
# loaded["TextBlob_polarity"] = 0.0
# loaded["TextBlob_subjectivity"] = 0.0
# loaded["Textblob_likely_party"] = 0

# loaded = loaded.apply(analyze_sentiment_textblob, axis=1)
# # %%
# print(loaded.columns)
# nue,dem,rep = 0,0,0
# for i in loaded["likely_party"]:
#     if i == 0:
#         nue += 1
#     elif i == 1:
#         dem += 1
#     else:
#         rep += 1
# print(nue, dem, rep)
# print(nue/len(loaded), dem/len(loaded), rep/len(loaded))
# # %%
# print(sum(loaded["polarity"])/len(loaded))
# print(min(loaded["polarity"]), max(loaded["polarity"]))

# # %%
