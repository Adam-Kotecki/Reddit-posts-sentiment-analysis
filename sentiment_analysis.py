import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

file_path = 'posts.xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

# The VADER (Valence Aware Dictionary and sEntiment Reasoner)
# lexicon is specifically tailored to sentiment analysis in social media texts, 
# like tweets, Facebook posts, and online reviews. It takes into account the nuances of social media language, 
# including emoticons, slang, and capitalization, to accurately gauge sentiment.

# Initialize NLTK sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

df['title_negative'] = ''
df['title_neutral'] = ''
df['title_positive'] = ''
df['title_overall'] = ''

for index, row in df.iterrows():
    title_text = row['title']
    title_sentiments = sia.polarity_scores(title_text)
    title_negative, title_neutral, title_positive, title_overall = title_sentiments.values()
    # Filling in sentiment columns:
    df.at[index, 'title_negative'] = title_negative
    df.at[index, 'title_neutral'] = title_neutral
    df.at[index, 'title_positive'] = title_positive
    df.at[index, 'title_overall'] = title_overall
    
def interpret_sentiment(score):
    if score > 0.2:
        return 'positive'
    elif score < -0.2:
        return 'negative'
    else:
        return 'neutral'
    
df['sentiment'] = df['title_overall'].apply(interpret_sentiment)

file_path = 'sentiment_analysis.xlsx'
df.to_excel(file_path, index=False)


    
    