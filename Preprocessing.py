"""df Includes all the tweets from The US
filtered_tweets_trump includes all the tweets that mention Trump but not Biden
filtered_tweets_biden includes all the tweets that mention Biden but not Trump
filtered_tweets includes all the tweets that mention either Trump or Biden but not both
"""
import pandas as pd
import os
import re

path = os.getcwd()
path_trump = path + "\\data\\hashtag_donaldtrump.csv"
trump = pd.read_csv(path_trump, lineterminator="\n")
path_biden = path + "\\data\\hashtag_joebiden.csv"
biden = pd.read_csv(path_biden, lineterminator="\n")
trump["source"] = "Trump"
biden["source"] = "Biden"
# Concatenate and remove duplicates
df = pd.concat([trump, biden], ignore_index=True)
df = df.drop_duplicates()

# Replace URLs with a placeholder text
df["tweet"] = df["tweet"].apply(lambda x: re.sub(r'http\S+', '[]', x))
# Drop duplicates based on the cleaned tweet text
df = df.drop_duplicates(subset=["tweet"])




us_states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "District of Columbia", "Florida", 
             "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", 
             "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", 
             "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", 
             "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]

df =df[df["state"].isin(us_states)]
print(len(df))


frequent_names_trump = [
    "Trump",
    "Donald" "Donald Trump",
    "@realDonaldTrump",
    "The Donald",
    "45",
    "Donald J. Trump",
    "DJT",
    "The Trump Administration",
    "Trumpster",
    "POTUS",
    "@POTUS",
    "Republican",
    "Republicans",
    "GOP",
    "MAGA",
    "Right Wing",
    "the Right",

]
frequent_names_biden = [
    "Biden",
    "Joe Biden",
    "@JoeBiden",
    "The Biden",
    "46",
    "Joseph R. Biden",
    "JRB",
    "The Biden Administration",
    "Bidenster",
    "Joe",
    "Joseph",
    "Joseph Biden",
    "Sleepy Joe",
    "Uncle Joe",
    "Dems",
    "Democrat",
    "Democrats",
    "Left Wing",
    "The Left",
]
pattern_trump = "|".join(frequent_names_trump)
pattern_biden = "|".join(frequent_names_biden)

# Create boolean masks where tweets contain any of the frequent names
mask_trump = df["tweet"].str.contains(pattern_trump, case=False, na=False)
mask_biden = df["tweet"].str.contains(pattern_biden, case=False, na=False)

# Combine the masks to filter for tweets containing Trump names but not Biden names or vice versa
testdf = df.copy()
testdf["tweet_about"] = "None"
filtered_tweets_trump = df[mask_trump & ~mask_biden]
filtered_tweets_biden = df[mask_biden & ~mask_trump]

testdf.loc[mask_trump & ~mask_biden, 'tweet_about'] = 'trump'
testdf.loc[mask_biden & ~mask_trump, 'tweet_about'] = 'biden'

filtered_tweets = testdf[(mask_trump & ~mask_biden) ^ (mask_biden & ~mask_trump)]
filtered_tweets = filtered_tweets.drop_duplicates(subset=["tweet"])

#%%
print(len(filtered_tweets))