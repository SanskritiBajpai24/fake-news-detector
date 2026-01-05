import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Setup & Cleaning
import nltk
nltk.download('stopwords')
stop_words = stopwords.words('english')
ps = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower().split()
    text = [ps.stem(word) for word in text if not word in stop_words]
    return ' '.join(text)

# 2. Data Preparation
# Load your datasets (assuming CSVs are in the same folder)
print("Loading data...")
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')
true_df['label'] = 0
fake_df['label'] = 1
df = pd.concat([true_df, fake_df]).sample(frac=1).reset_index(drop=True)

print("Cleaning text (this may take a minute)...")
df['clean_text'] = df['text'].apply(clean_text)

# 3. Naive Bayes Path
print("Training Naive Bayes...")
tfidf = TfidfVectorizer(max_features=5000)
X_nb = tfidf.fit_transform(df['clean_text']).toarray()
y = df['label']
X_train_nb, X_test_nb, y_train, y_test = train_test_split(X_nb, y, test_size=0.2)

nb_model = MultinomialNB()
nb_model.fit(X_train_nb, y_train)

# Save NB components
pickle.dump(nb_model, open('nb_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))

# 4. LSTM Path
print("Training LSTM...")
max_vocab = 10000
max_len = 300
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(df['clean_text'])
X_lstm = pad_sequences(tokenizer.texts_to_sequences(df['clean_text']), maxlen=max_len)

X_train_ls, X_test_ls, y_train_ls, y_test_ls = train_test_split(X_lstm, y, test_size=0.2)

lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_vocab, 128, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_ls, y_train_ls, epochs=3, batch_size=64, validation_split=0.1)

# Save LSTM components
lstm_model.save('lstm_model.h5')
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))

print("All models saved successfully!")