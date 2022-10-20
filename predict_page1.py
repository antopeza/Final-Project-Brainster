import streamlit as st
import pickle
import numpy as np
import pandas as pd
#import sklearn
#import ipynb
#from bow_rf_lr_ver1 import clean_sentence,get_cleaned_senteces,to_df

# making list from entries
def get_data():
    return []

def load_model():
    with open('saved_steps1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
 #data = {"model": regressor, "get_cleaned_senteces": get_cleaned_senteces, "to_df": to_df,"clean_sentence": clean_sentence}
   # dali get_cleaned_senteces so parametri ili bez parametri vo pkl 
data = load_model()

regressor = data["model"]
get_cleaned_senteces = data["get_cleaned_senteces"]
to_df = data["to_df"]
clean_sentence = data["clean_sentence"]

def show_predict_page():
    st.title("Similarity Prediction")

    st.write("""### We need 2 sentences to predict similarity""")

    input1 = st.text_input("question1")
    input2 = st.text_input("question2")

    ok = st.button("Calculate Similarity")
    if ok:
        # Create df from inpu1 and input 2
        df =  pd.DataFrame(get_data().append({"question1": input1, "question2": input2}))
        
        # Transform dataframe, Without stopwords and stem, lemmatized
        q1_without_sw_l, q2_without_sw_l = get_cleaned_senteces(df, stopwords = True, lemmatize = True, stem = False)
        X_without_sw_l = to_df(q1_without_sw_l, q2_without_sw_l, df)
        
        #Prediction
        similarity = regressor.predict(X_without_sw_l)
        st.subheader(f"The estimated similarity is ${similarity[0]:.2f}")
        #st.subheader(f"The estimated similarity is 0.5")

