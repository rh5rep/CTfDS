#%%
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from Preprocessing import filtered_tweets_biden, filtered_tweets_trump, filtered_tweets
from process_tweet import process_tweet
#%%
class PoliticalSentimentAnalyzer:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        
        # Add political booster words
        self.sid.lexicon.update({
            'lock': -2.0,
            'corrupt': -3.0,
            'fraud': -3.0,
            'stolen': -3.0,
            'rigged': -3.0,
            'illegal': -2.0,
            'fake': -2.0,
            'socialist': -2.0,
            'communist': -2.0,
            'radical': -2.0,
            'victory': 2.0,
            'win': 2.0,
            'winner': 2.0,
            'support': 2.0,
            'strong': 2.0,
            'leader': 2.0,
        })
        
        # Patterns that modify sentiment
        self.negative_patterns = [
            (r"not my president", -0.5),
            (r"lock .* up", -0.6),
            (r"stop the steal", -0.7),
            (r"(never|not) (support|trust|believe)", -0.5),
        ]
        
        self.positive_patterns = [
            (r"(best|great|excellent) president", 0.5),
            (r"make .* great again", 0.6),
            (r"god bless", 0.4),
            (r"(proud|happy|glad) to support", 0.5),
        ]
    
    def check_patterns(self, text):
        modifier = 0
        
        for pattern, score in self.negative_patterns:
            if re.search(pattern, text.lower()):
                modifier += score
                
        for pattern, score in self.positive_patterns:
            if re.search(pattern, text.lower()):
                modifier += score
                
        return modifier

    def analyze_tweet(self, row):
        tweet = process_tweet(row["tweet"])  # Using your existing process_tweet function
        
        # Get base VADER sentiment
        sentiment_dict = self.sid.polarity_scores(tweet)
        
        # Apply pattern modifiers
        pattern_modifier = self.check_patterns(tweet)
        modified_compound = sentiment_dict['compound'] + pattern_modifier
        
        # Ensure compound stays in [-1, 1] range
        modified_compound = max(min(modified_compound, 1.0), -1.0)
        
        # Update row with sentiment scores
        row['positive'] = sentiment_dict['pos']
        row['neutral'] = sentiment_dict['neu']
        row['negative'] = sentiment_dict['neg']
        row['compound'] = modified_compound
        
        # Determine likely party with adjusted thresholds
        if abs(modified_compound) < 0.1:  # Reduced neutral threshold
            row['likely_party'] = 0
        elif row["tweet_about"] == "biden":
            row['likely_party'] = 1 if modified_compound > 0 else 2
        elif row["tweet_about"] == "trump":
            row['likely_party'] = 2 if modified_compound > 0 else 1
        
        return row

# Usage with your existing code structure
analyzer = PoliticalSentimentAnalyzer()
filtered_tweets = filtered_tweets.apply(analyzer.analyze_tweet, axis=1)

# Test a specific tweet
def test_tweet(text, tweet_about="biden"):
    test_row = {"tweet": text, "tweet_about": tweet_about}
    result = analyzer.analyze_tweet(test_row)
    print(f"Tweet: {text}")
    print(f"Compound Score: {result['compound']:.3f}")
    print(f"Likely Party: {result['likely_party']}")
    return result

# Example test
test_tweet("Biden is not my president")
test_tweet("lock him up")
test_tweet("Trump made America great again", "trump")
#%%
print(filtered_tweets.columns)
print(filtered_tweets[["tweet", "compound", "likely_party"]].head())
#%%
import pickle as pkl
with open('filtered_tweetsC.pkl', 'wb') as f:
    pkl.dump(filtered_tweets, f)
# %%
print(filtered_tweets[filtered_tweets["likely_party"] == 0]/len(filtered_tweets), filtered_tweets[filtered_tweets["likely_party"] == 1]/len(filtered_tweets), filtered_tweets[filtered_tweets["likely_party"] == 2]/len(filtered_tweets))
# %%
nue,dem,rep = 0,0,0
for i in filtered_tweets["likely_party"]:
    if i == 0:
        nue += 1
    elif i == 1:
        dem += 1
    else:
        rep += 1
print(nue/len(filtered_tweets), dem/len(filtered_tweets), rep/len(filtered_tweets))
# %%
