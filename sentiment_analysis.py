from process_tweet import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Sentiment analysis

def analyze_sentiment(text):
    # Create a SentimentIntensityAnalyzer object
    analyzer = SentimentIntensityAnalyzer()

    text = process_tweet(text)
    
    # Analyze the sentiment for the text
    sentiment = analyzer.polarity_scores(text)

    # Get the compound score
    # compound = sentiment['compound']
    
    # Return the sentiment
    return sentiment

# Example usage
# text = "The US is the Best Country in the world! #USA #US #America!! #UnitedStates!"
# text = "The US is the worst Country in the world! #USA #US #America!! #UnitedStates!"
text = "Trump is the best man for the job! #Trump2020 #MAGA"
sentiment = analyze_sentiment(text)
print(sentiment)
