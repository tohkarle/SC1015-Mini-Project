# Predicting Disasters with Tweets

## About
This is a "Getting Started" kaggle competition for NLP which we have chosen as our SC1015 Mini Project.
<br>The order of the python notebooks are as follow:

1. [Data Preprocessing](https://github.com/tohkarle/SC1015-mini-project/blob/main/preprocessing)
2. [Converting to Text Embeddings](https://github.com/tohkarle/SC1015-mini-project/tree/main/word_embeddings)  
3. [Traditional Model Training and Evaluation](https://github.com/tohkarle/SC1015-mini-project/tree/main/traditional_models)
4. [Transformer Model Training and Evaluation](https://github.com/tohkarle/SC1015-mini-project/tree/main/transformer_models)
   
## Contributors

- Ler Hong @llerhong - Data Preprocessing, Logistic Regression, Research and Presentation
- Kar Le @tohkarle - Data Preprocessing, Text Embeddings, Gradient Boosting, GPT-2
- Yu Hao @yuhaopro - Data Preprocessing, Text Embeddings, Random Forest, BERT

## Problem Definition

Given the significant role Twitter plays as a crucial communication channel during emergencies, it is a platform where people can provide real-time updates on disasters as they unfold. As a result, various agencies, including disaster relief organizations and news agencies, are keen on monitoring Twitter for relevant information automatically. However, accurately predicting a real disaster tweet can prove challenging in practice, because some tweets may contain misleading words or phrases that a machine may mistake for a disaster. 

## Embedding Models

- The file naming convention is as follows: [Embedding Model][Vector Dimension][Average Embedding for OOV]
- Eg. `glove_50_0v.csv` implies it uses GloVe embedding with a vector dimension of 50 and OOV words are replaced with 0 vectors.
#### Note:
- fasttext uses subword information so it does not have a tag for how it calculates average embedding.
- For word2vec, the dataset was used to generate custom embedding through word2vec, so there are no OOV words.

1. GloVe
3. Fasttext 
5. Word2vec

## Traditional Models 

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting

## Transformer Models

#### Note: There are 3 files for BERT each using 3 different approaches while leveraging on the pre-trained BERT model. 
    
1. BERT
    - BERT_BFSC.ipynb
        - This file uses `BertForSequenceClassifier` and `Trainer` modules from transformers library. 
    - BERT_numerical.ipynb
        - This file combines `text feature` and `numerical features` using a Simple Feed Foward Neural Network.
    - BERT_CC.ipynb
        - This file explores using text feature by itself, and concatenating text feature with numericals into 1 text feature
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
4. How to initialize and train traditional and transformer models with k-fold cross validation
5. Learn how to utilize `transformers` and `scikit-learn` library
6. Hyperparameter tuning with Random Search, Grid Search and Bayesian Optimization

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
- https://arxiv.org/pdf/1706.03762.pdf



