import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment of a text
def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

# Example usage
text = "Fairly good, though it took a bit longer than anticipated to get everything right. The project had its ups and downs, but overall, it was a success."
sentiment_scores = analyze_sentiment(text)
print(sentiment_scores)
