import streamlit as st
import pandas as pd
import numpy as np
from numpy import random
import os 
import re
import string
from string import punctuation
#import gensim
#from gensim.parsing.preprocessing import remove_stopwords
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


## Function streamlit 
def main():
    st.write(""" 
    # SIMPLE TEXT SIMILARITY APP 
    Compare two sentences and check it out!
    """)

    col1, col2 = st.columns(2)
    with col1:
        sentence_1 = st.text_area('Please write your sentence 1')     
        sentence_1 = sentence_1.lower().strip()
        sentence_1 = re.sub(r'[^a-z0-9\s]', '', sentence_1)  # removing all caracters that are not alpha numeric
        sentence_1 = sentence_1.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        
        #st.success(sentence_1
        #sentence_1 = remove_stopwords(sentence_1)
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sentence_1)
        arr = X.toarray()
        heatmap = create_heatmap(cosine_similarity(arr))
        
        
        st.success(heatmap)
                   
    with col2:
        sentence_2 = st.text_area('Please write your sentence 1')
        sentence_2 = sentence_2.lower().strip()
        sentence_2 = re.sub(r'[^a-z0-9\s]', '', sentence_2)  # removing all caracters that are not alpha numeric
    
        sentence_2 = sentence_2.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        #sentence_1 = remove_stopwords(sentence_1)
        
        st.success(sentence_2)
        
    



                                 
    
if __name__== '__main__':
    main()
