import streamlit as st
import re
import pickle
import string 
from nltk.corpus import stopwords
import nltk 
from nltk.stem.porter import PorterStemmer
pt=PorterStemmer()

def preprocess_text(text):
    # convert to lowercase
    text = text.lower()
    
    # remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # lemmatize the tokens
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # join the tokens back into a single string with spaces
    preprocessed_text = ' '.join([token + ' ' for token in lemmatized_tokens])
    
    return preprocessed_text

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")
input_email = st.text_area("Enter the message")
if st.button('Predict'):

    # 1. preprocess
    transformed_sms = preprocess_text(input_email)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")