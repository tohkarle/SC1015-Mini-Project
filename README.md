# Predicting Disasters with Tweets

## About
This is a "Getting Started" kaggle competition for NLP which we have chosen as our SC1015 Mini Project.
With the recent advent of disruptive technologies employing state of the art NLP techniques such as chatGPT, we are looking to explore the difference which traditional and transformer models differ in performance, and how it can apply to real world applications such as predicting disasters with tweets.
<br>The order of the python notebooks are as follow:

1. [Data Preprocessing](https://github.com/tohkarle/SC1015-mini-project/blob/main/data_preprocessing.ipynb)
    - Data cleaning
    - Text mining
    - Keyword Feature 
    - Location Feature 
    - Target Distribution 
    - Meta Features 
2. [Converting to Text Embeddings](https://github.com/tohkarle/SC1015-mini-project/tree/main/word_embeddings)  
    - [Fasttext (Pre-trained)](https://github.com/tohkarle/SC1015-mini-project/blob/main/word_embeddings/glove_fasttext.ipynb)
    - [GloVe (Pre-trained)](https://github.com/tohkarle/SC1015-mini-project/blob/main/word_embeddings/glove_fasttext.ipynb)
    - [Word2vec (Trained on Dataset)](https://github.com/tohkarle/SC1015-mini-project/blob/main/word_embeddings/word2vec.ipynb)
    - [GloVe + Word2vec (Pre-trained + OOV replaced with custom embeddings)](https://github.com/tohkarle/SC1015-mini-project/blob/main/word_embeddings/glove_fasttext.ipynb)
3. [Traditional Model Training and Evaluation](https://github.com/tohkarle/SC1015-mini-project/tree/main/traditional_models)
    - [Logistic Regression](https://github.com/nicklimmm/movie-analysis/blob/main/traditional_models/logistic-regression.ipynb)
    - [Random Forest Classifier](https://github.com/tohkarle/SC1015-mini-project/blob/main/traditional_models/random-forest.ipynb)
    - [Gradient Boosting](https://github.com/tohkarle/SC1015-mini-project/blob/main/traditional_models/gradient_boosting.ipynb)
4. [Transformer Model Training and Evaluation](https://github.com/tohkarle/SC1015-mini-project/tree/main/transformer_models)
    - [BERT](https://github.com/nicklimmm/movie-analysis/blob/main/transformer_models/BERT.ipynb)
    - [GPT-2](https://github.com/nicklimmm/movie-analysis/blob/main/transformer_models/GPT2.ipynb)
    
## Contributors

- Ler Hong @llerhong - Data Preprocessing, Logistic Regression, Research and Presentation
- Kar Le @tohkarle - Data Preprocessing, Text Embeddings, Gradient Boosting, GPT-2
- Yu Hao @yuhaopro - Data Preprocessing, Text Embeddings, Random Forest, BERT

## Problem Definition

Given the significant role Twitter plays as a crucial communication channel during emergencies, it is a platform where people can provide real-time updates on disasters as they unfold. As a result, various agencies, including disaster relief organizations and news agencies, are keen on monitoring Twitter for relevant information automatically. However, accurately predicting a real disaster tweet can prove challenging in practice, because some tweets may contain misleading words or phrases that a machine may mistake for a disaster. 

## Embedding Models

Converting text into embeddings helps to represent words in a high-dimensional space, such that the relationships between words are preserved. This step is necessary as it allows us to use the vector representation of a word as a feature in our traditional machine learning models. We will take the average embedding derived from each tweet.
- The file naming convention is as follows: [Embedding Model]_[Vector Dimension]_[Average Embedding for OOV]
- Eg. glove_50_0v.csv implies it uses GloVe embedding with a vector dimension of 50 and OOV words are replaced with 0 vectors.
#### Note:
- fasttext uses subword information so it does not have a tag for how it calculates average embedding.
- For word2vec, the dataset was used to generate custom embedding through word2vec, so there are no OOV words.

1. GloVe
    - glove_50d_0v.csv
    - glove_50d_custom.csv
3. Fasttext
    - fasttext-300d.csv 
5. Word2vec
    -  word2vec_50d.csv

## Traditional Models 

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting

## Transformer Models

1. BERT
2. GPT-2

## Conclusion

1. GPT-2 performed the best out of all the models that were chosen (BERT was a close second).
2. Transformer models performed generally better than Traditional models with text data.
3. The choice of embedding models and traditional models did not significantly impact the final accuracy score.
4. Text Feature was the most important predictor among all features.

## Reasoning
1. Text mining and converting to text embeddings is a pre-requisite for traditional models, and the process could have resulted in loss of information. This pales in comparison to transformer models that can do the entire process by itself using the encoding and decoding mechanism.
2. The self-attention mechanism of Transformers could have resulted in more accurate text embeddings which better capture the semantics of the raw text itself.

## Skills Acquired
1. Text Mining techniques
2. Feature Extraction such as meta features and Target Encoding
3. How to use text embeddings to generate meaningful information out from raw text
4. How to initialize and train traditional and transformer models with cross-fold validation
5. Hyperparameter tuning with Random Search, Grid Search and Bayesian Optimization

## References
- https://www.kaggle.com/competitions/nlp-getting-started/overview
- https://towardsdatascience.com/a-guide-to-word-embeddings-8a23817ab60f
- https://nlp.stanford.edu/projects/glove/
- https://www.tensorflow.org/tutorials/text/word2vec
- https://fasttext.cc/docs/en/support.html
- https://xgboost.readthedocs.io/en/stable/
- https://huggingface.co/docs/transformers/model_doc/bert
- https://huggingface.co/docs/transformers/model_doc/gpt2
- https://huggingface.co/docs/evaluate/a_quick_tour
- https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/trainer
- https://scikit-learn.org/stable/



