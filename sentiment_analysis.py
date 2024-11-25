#%%
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
# text = "MAGA are clowns"
text = """
#TRUMP will Do AnYTHING to #liecheatsteal #votes @realDonaldTrump is a #CrookedTrump #liar #DerangedDonald []"""
sentiment = analyze_sentiment(text)
print(sentiment)

# #%%
# import spacy
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from process_tweet import *  # Assuming this is a custom module for preprocessing tweets

# # Initialize spaCy
# nlp = spacy.load("en_core_web_sm")

# # Preprocessing function using spaCy
# def preprocess_text(text):
#     # Process the text with spaCy NLP pipeline
#     doc = nlp(text)

#     # Remove stop words and non-alphabetical characters, lemmatize words
#     processed_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

#     return processed_text

# # Sentiment analysis
# def analyze_sentiment(text):
#     # Create a SentimentIntensityAnalyzer object
#     analyzer = SentimentIntensityAnalyzer()

#     # Preprocess the tweet text using spaCy
#     text = preprocess_text(text)
    
#     # Further process the text using your process_tweet function (if necessary)
#     text = process_tweet(text)  # Assuming this does additional custom processing
    
#     # Analyze the sentiment for the text
#     sentiment = analyzer.polarity_scores(text)

#     return sentiment

# # Example usage
# text = "MAGA are clowns"
# sentiment = analyze_sentiment(text)
# print(sentiment)

# # %%
# import spacy
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# # Load the English language model
# nlp = spacy.load("en_core_web_lg")

# # Create a SentimentIntensityAnalyzer object
# analyzer = SentimentIntensityAnalyzer()

# def analyze_sentiment(text):
#     # Process the text using spaCy
#     doc = nlp(text)
    
#     # Get the sentiment score from spaCy
#     spacy_sentiment = doc.sentiment
    
#     # Get the sentiment score from VADER
#     vader_sentiment = analyzer.polarity_scores(text)
    
#     # Combine the scores
#     overall_sentiment = (spacy_sentiment + vader_sentiment['compound']) / 2
    
#     return overall_sentiment

# # Example usage
# text = "Trump for president!"
# sentiment = analyze_sentiment(text)
# print(f"Sentiment score: {sentiment}")


# # %%
# # Load the large English language model
# import spacy
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from process_tweet import process_tweet
# nlp = spacy.load("en_core_web_lg")

# # Create a SentimentIntensityAnalyzer object
# analyzer = SentimentIntensityAnalyzer()

# def analyze_sentiment_with_context(text):
#     # Process the text using spaCy
#     text = process_tweet(text)
#     doc = nlp(text)
#     print(doc)
    
#     # Extract the context from the sentence
#     context = ' '.join([token.text for token in doc if token.dep_ in ('nsubj', 'dobj', 'pobj', 'ROOT')])
    
#     # Get the sentiment score from VADER for the context
#     vader_sentiment = analyzer.polarity_scores(context)
    
#     return vader_sentiment

# # Example usage
# text = """
# "It is demonstrated that with #trump we are on the threshold of a #supremacist #genocide that precedes a world war

# #AllVotesMatter #AllVotesMustBeCounted #CountEveryVote #USA #dictatorship #SaveDemocracy"""
# sentiment = analyze_sentiment_with_context(text)
# print(f"Sentiment score: {sentiment}")
# %%

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from process_tweet import process_tweet
nlp = spacy.load("en_core_web_lg")

# Create a SentimentIntensityAnalyzer object
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_with_context(text):
    # Process the text using spaCy
    text = process_tweet(text)
    doc = nlp(text)
    print(doc)
    
    # Extract the context from the sentence
    context = ' '.join([token.text for token in doc if token.dep_ in ('nsubj', 'dobj', 'pobj', 'ROOT')])
    
    # Get the sentiment score from VADER for the context
    vader_sentiment = analyzer.polarity_scores(context)
    
    return vader_sentiment

# Example usage
# text = """
# "It is demonstrated that with #trump we are on the threshold of a #supremacist #genocide that precedes a world war

# #AllVotesMatter #AllVotesMustBeCounted #CountEveryVote #USA #dictatorship #SaveDemocracy"""
text ="""
#TRUMP will Do AnYTHING to #liecheatsteal #votes @realDonaldTrump is a #CrookedTrump #liar #DerangedDonald []
# """
sentiment = analyze_sentiment_with_context(text)
print(f"Sentiment score: {sentiment}")
# %%
