import streamlit as st
import pandas as pd

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

    
    click = st.button('click me')
    
    
