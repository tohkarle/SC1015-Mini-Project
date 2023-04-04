import numpy as np
import pandas as pd
import ast

def build_vocab(df_column):
    vocab = set()
    for token_list_string in df_column:
        tokens = ast.literal_eval(token_list_string)
        for token in tokens:
            print(token)
            vocab.add(token)
    return vocab

def missing_words(vocab, embeddings):
    # Check if each word in your vocabulary is in the GloVe embeddings
    missing_words = []
    for word in vocab:
        if word not in embeddings:
            missing_words.append(word)
    # Print the number of missing words and the missing words themselves
    print(f"Number of missing words: {len(missing_words)}")
    print(f"Missing words: {missing_words}")

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def average_embeddings_glove(tokens, embeddings, embedding_dim):
    token_vectors = [embeddings.get(token, np.zeros(embedding_dim)) for token in tokens]
    avg_vector = np.mean(token_vectors, axis=0)
    return avg_vector

def average_embeddings_fasttext(tokens, fasttext_model):
    token_vectors = []
    for token in tokens:
        try:
            vector = fasttext_model.get_vector(token)
            token_vectors.append(vector)
        except KeyError:
            pass  # Ignore words not in the FastText vocabulary
    avg_vector = np.mean(token_vectors, axis=0)
    return avg_vector
