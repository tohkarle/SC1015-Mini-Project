# This file contains all the necessary functions for using text mining techniques on the data

import pandas as pd
import numpy as np
import re
import nltk

from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk import ngrams
from nltk import pos_tag
from nltk import ne_chunk
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')


def convert_to_lower(tweets):
    lower_tweets = []
    for tweet in tweets:
        lower_tweets.append(tweet.lower())
    return lower_tweets

def remove_special_chars_urls_mentions(tweets):
    processed_tweets = []
    for tweet in tweets:
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'@\w+', '', tweet)
        tweet = re.sub(r'\W', ' ', tweet)
        processed_tweets.append(tweet)
    return processed_tweets

def tokenize_tweets(tweets):
    tokenized_tweets = []
    for tweet in tweets:
        tokens = word_tokenize(tweet)
        tokenized_tweets.append(tokens)
    return tokenized_tweets

def correct_spelling(tweets):
    spell = SpellChecker()
    corrected_tweets = []
    for tokens in tweets:
        corrected_tokens = []
        for token in tokens:
            corrected_token = spell.correction(token)
            if corrected_token:
                corrected_tokens.append(corrected_token)
            else:
                corrected_tokens.append(token)
        corrected_tweets.append(corrected_tokens)
    return corrected_tweets

def remove_stopwords(tweets):
    stop_words = set(stopwords.words('english'))
    filtered_tweets = []
    for tokens in tweets:
        filtered_tokens = [token for token in tokens if token not in stop_words]
        filtered_tweets.append(filtered_tokens)
    return filtered_tweets

def lemmatize_words(column):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []

    for tokens in column:
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_words.append(lemmatized_tokens)

    return lemmatized_words


def generate_ngrams(tweets, n=2):
    ngram_tweets = []
    for tokens in tweets:
        token_ngrams = list(ngrams(tokens, n))
        ngram_tweets.append(token_ngrams)
    return ngram_tweets

def pos_tagging(tweets):
    pos_tagged_tweets = []
    for tokens in tweets:
        tagged_tokens = pos_tag(tokens)
        pos_tagged_tweets.append(tagged_tokens)
    return pos_tagged_tweets

def named_entity_recognition(column):
    ner_results = []

    for pos_tags in column:
        ne_tree = ne_chunk(pos_tags)
        entities = [f"{tree[0][0]} ({tree.label()})" for tree in ne_tree if hasattr(tree, 'label')]
        ner_results.append(entities)

    return ner_results
