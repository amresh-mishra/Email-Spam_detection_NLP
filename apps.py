from flask import Flask, render_template, request
import re
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

pt = PorterStemmer()

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_email = request.form['email']

        # Load the trained model and vectorizer
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))

        # Preprocess the input email
        transformed_sms = preprocess_text(input_email)

        # Vectorize the preprocessed email
        vector_input = tfidf.transform([transformed_sms])

        # Predict using the loaded model
        result = model.predict(vector_input)[0]

        if result == 1:
            prediction = "Spam"
        else:
            prediction = "Not Spam"

        return render_template('index.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
