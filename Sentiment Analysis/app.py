from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX_NUM_WORDS = 100000
review = pd.read_csv("yelp.csv")
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
tok = Tokenizer(num_words=MAX_NUM_WORDS)
tok.fit_on_texts(review.text.values)
MAX_SEQ_LEN=200
app = Flask(__name__)

model=load_model('model.h5', compile = False)


@app.route('/')
def hello_world():
    return render_template("sentiment.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    sample_review=np.array(list(request.form.values()))

    print(sample_review)
    test_tokens = tok.texts_to_sequences(sample_review)
    print(test_tokens)
    # tokenize text
    test_tokens = tok.texts_to_sequences(sample_review)
    test_tokens = pad_sequences(test_tokens, maxlen=MAX_SEQ_LEN,
                                dtype='int32', padding='pre', truncating='pre',
                                value=0.0)
    test_tokens = np.array(test_tokens)
    print(test_tokens.shape)
    prediction=model.predict(test_tokens)
    print("Prediction is",prediction)
    output='{0:.{1}f}'.format(prediction[0][0], 2)

    if output>str(0.5):
        return render_template('sentiment.html',pred='You got a positive review.\nProbability of sentiment is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('sentiment.html',pred='You got a negative review.\n Probability of sentiment is {}'.format(output),bhai="Your Forest is Safe for now")


if __name__ == '__main__':
    app.run(debug=True)
