import streamlit as st
import pandas as pd
import numpy as np
from numpy import random
import os 
import re
import string
from string import punctuation

# #from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# #import sklearn.metrics as metrics
# #from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# import nltk
# from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer
# from nltk.corpus import wordnet, stopwords
# from nltk import pos_tag, word_tokenize

# import gensim
# from gensim.parsing.preprocessing import remove_stopwords

# #from keras.preprocessing.text import Tokenizer
# #from keras_preprocessing.sequence import pad_sequences
# #from keras.models import Model, Sequential, load_model
# #from keras.layers import Input, Embedding, LSTM, Dense
# #from keras.callbacks import ModelCheckpoint, EarlyStopping
# #from scikeras.wrappers import KerasClassifier

# import pickle
# from pickle import dump
# from pickle import load

# import warnings
# warnings.filterwarnings('ignore')

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")
# nltk.download('omw-1.4')

# @st.cache
# def get_wordnet_pos(tag):
#     if tag.startswith("N"):
#         return wordnet.NOUN
#     if tag.startswith("J"):
#         return wordnet.ADJ
#     if tag.startswith("V"):
#         return wordnet.VERB
#     if tag.startswith("R"):
#         return wordnet.ADV
#     return wordnet.NOUN

# #Function Pre-processing
# @st.cache
# def clean_sentence(sentence):
#     #sentence = []
#     sentence = sentence.lower().strip()
#     sentence = re.sub(r'[^a-z0-9\s]', '', sentence)  # removing all caracters that are not alpha numeric
#     #sentence = re.sub(r'\s{2,}', ' ', sentence
#     # remove punctuation
#     sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    
#     sentence = remove_stopwords(sentence)  # stopwords are adding noises, and some algoritham can not handle 
#     return sentence 
# #     sent_lemma = ''
# #     for word, tag in pos_tag(word_tokenize(sentence)):
# #             wntag = tag[0].lower()
# #             wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
# #             sent_lemma += ' ' + lemmer.lemmatize(word, wntag) if wntag else word
# #             sentence = sent_lemma
# #     return sentence





## Function streamlit 
def main():
    st.write(""" 
    # SIMPLE TEXT SIMILARITY APP 
    Compare two sentences and check it out!
    """)

    col1, col2 = st.columns(2)
    with col1:
        word_1 = st.text_area('Please write your sentence 1')
    with col2:
        word_2 = st.text_area('Please write your sentence 2')
        
        #click = st.button('click me')
        
        if st.button('click me'):
            word_1_m = [sent.lower().split(" ") for sent in word_1]
            word_2_m = [sent.lower().split(" ") for sent in word_2]
        
            return word_1_m, word_2_m
        
        #else: 
         #   st.write('try again')
        
    
    
    
#     sentences = ["The bottle is empty",
# "There is nothing in the bottle"]
# sentences = [sent.lower().split(" ") for sent in sentences]
# jaccard_similarity(sentences[0], sentences[1])
    
    
#     if st.button('Compare text'):
#         clean_1 = clean_sentence(word_1)
#         #clean_2 = clean_sentence(word_2
#         #summary_result = clean_1 + clean_2
        
#         st.success(clean_1)
                                 
    
if __name__== '__main__':
    main()
