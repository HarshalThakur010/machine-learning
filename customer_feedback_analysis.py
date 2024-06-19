import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from collections import defaultdict
from jellyfish import jaro_winkler_similarity
from spacy.tokens import Doc
from spacy.matcher import Matcher

#import spacy.cli
#spacy.cli.download("en_core_web_lg")

nlp = spacy.load("en_core_web_lg")

# Load data
df = pd.read_excel('wetransfer_campaign_data-csv_2024-04-13_0433/CustFeedback.xlsx')
df["Feedback"] = df["Feedback"].astype(str)
df["Feedback"] = df["Feedback"].fillna('')
df_sample = pd.concat([df.head(100),df.tail(100)])

# Step 1: Data Cleaning
def clean_text(text):
    text = re.sub(r'[^\w\s.]', '', text)  # Remove special characters except for .
    text = re.sub(r'\s{2,}', ' ', text).strip()  # Replace multiple spaces with a single space and trim leading/trailing spaces
    text = re.sub(r'\.{2,}', '.', text)
    return text

df['Feedback_cleaned'] = df['Feedback'].apply(lambda x: clean_text(str(x)) if pd.notnull(x) else '')
df_sample = pd.concat([df.head(100),df.tail(100)])


# Step 2: Tokenization and POS Tagging
def pos_tagging(text):
    tags = []
    doc = nlp(text)
    for token in doc:
        tags.append([str(token), token.pos_])
    return tags


df_sample['Doc'] = df_sample['Feedback_cleaned'].apply(pos_tagging)


# Step 3: Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if not text or len(text)<=2:
        return 'neutral'
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['Sentiment'] = df['Feedback_cleaned'].apply(get_sentiment)
df_sample2 = pd.concat([df.head(100),df.tail(100)])


# Step 4: Extract Summary Phrases
def extract_summary_phrases(pos_tags):
    phrases = []
    for i in range(len(pos_tags) - 1):
        word, pos = pos_tags[i]
        next_word, next_pos = pos_tags[i + 1]

        if pos in ['ADJ', 'VERB'] and next_pos in ['NOUN', 'PROPN']:
            phrases.append(f"{word} {next_word}")
        elif pos in ['ADJ', 'VERB'] and next_pos in ['DET', 'ADP']:
            if i + 2 < len(pos_tags):
                next_next_word, next_next_pos = pos_tags[i + 2]
                if next_next_pos in ['NOUN', 'PROPN']:
                    phrases.append(f"{word} {next_word} {next_next_word}")
    return phrases


df_sample['Phrases'] = df_sample['Doc'].apply(extract_summary_phrases)


# Step 5: Classify Similar Key-Phrases
def classify_phrases(phrases, threshold=0.85):
    classified_phrases = defaultdict(list)
    for phrase in phrases:
        found = False
        for key in classified_phrases.keys():
            if jaro_winkler_similarity(phrase, key) > threshold:
                classified_phrases[key].append(phrase)
                found = True
                break
        if not found:
            classified_phrases[phrase].append(phrase)
    return classified_phrases

all_phrases = [phrase for sublist in df_sample['Phrases'] for phrase in sublist]
classified_phrases = classify_phrases(all_phrases)
