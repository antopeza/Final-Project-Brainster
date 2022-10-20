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





## Function streamlit 
def main():
    st.write(""" 
    # SIMPLE TEXT SIMILARITY APP 
    Compare two sentences and check it out!
    """)

    col1, col2 = st.columns(2)
    with col1:
        sentence_1 = st.text_area('Please write your sentence 1')
        #word_1_m = [sent.lower().split(" ") for sent in word_1]
        
        
        sentence_1 = sentence_1.lower().strip()
        #sentence_1 = re.sub(r'[^a-z0-9\s]', '', sentence_1)  # removing all caracters that are not alpha numeric
    #sentence = re.sub(r'\s{2,}', ' ', sentence
   
        #sentence_1 = sentence_1.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        #sentence_1 = remove_stopwords(sentence_1)
        
        st.success(sentence_1)
#     with col2:
#         word_2 = st.text_area('Please write your sentence 2')
#         word_2_m = [sent.lower().split(" ") for sent in word_2]
        
        #st.success(word_1_m)
        
        #click = st.button('click me')
        
        #if st.button('click me'):
#             word_1_m = [sent.lower().split(" ") for sent in word_1]
#             word_2_m = [sent.lower().split(" ") for sent in word_2]
        
           # st.json(word_1_m, word_2_m)
        
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
