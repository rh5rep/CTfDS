import re
import wordninja

def process_tweet(text):
    def split_hashtag(match):
        words = wordninja.split(match.group())
        return " ".join(words)

    # Remove all URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Replace all hashtags with split words
    processed_text = re.sub(r'#\w+\b', split_hashtag, text)
    return processed_text

# Example usage
# text = "Check this out: https://example.com The US is the Best Country in the world! #USA #US #America!! #UnitedStates!"
# processed_text = process_tweet(text)
# print(processed_text)
