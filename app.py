import streamlit as st
import pickle
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Config
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Load Models
@st.cache_resource
def load_models():
    nb = pickle.load(open('nb_model.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    lstm = tf.keras.models.load_model('lstm_model.h5')
    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
    return nb, tfidf, lstm, tokenizer

nb_model, tfidf, lstm_model, tokenizer = load_models()
ps = PorterStemmer()

def clean_input(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    return ' '.join(text)

# UI Design
st.title("📰 NLP Fake News Detector")
st.write("This tool uses Machine Learning (Naive Bayes) and Deep Learning (LSTM) to verify news authenticity.")

user_text = st.text_area("Paste the news article text here:", height=250)

if st.button("Verify Authenticity"):
    if user_text.strip() == "":
        st.error("Please enter some text!")
    else:
        cleaned = clean_input(user_text)
        
        # NB Prediction
        vec_nb = tfidf.transform([cleaned]).toarray()
        prob_nb = nb_model.predict_proba(vec_nb)[0][1]
        
        # LSTM Prediction
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=300)
        prob_lstm = lstm_model.predict(padded)[0][0]
        
        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Naive Bayes Confidence", f"{prob_nb*100:.1f}%")
            if prob_nb > 0.5: st.error("Likely FAKE")
            else: st.success("Likely REAL")
            
        with col2:
            st.metric("LSTM Confidence", f"{prob_lstm*100:.1f}%")
            if prob_lstm > 0.5: st.error("Likely FAKE")
            else: st.success("Likely REAL")