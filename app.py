import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import ISRIStemmer
from langdetect import detect

# Arabic preprocessing functions
def clean_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Remove non-Arabic characters
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u0674\u06D6-\u06ED]', '', text)  # Remove Arabic diacritics
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text

def remove_stopwords(text , lang):

        stop_words = set(stopwords.words('arabic'))
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

        return ' '.join(filtered_tokens)



def stem_text(text):
    stemmer = ISRIStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

# TF-IDF Vectorizer
def tfidf_vectorizer(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


# Language detection
def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = None
    return lang

# Search function
def search(query, vectorizer, corpus, sentences):
    query = clean_text(query)
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, corpus).flatten()
    sorted_indices = cosine_similarities.argsort()[::-1]
    results = []
    for idx in sorted_indices:
        if cosine_similarities[idx] > 0:
            for sentence in sentences[idx]:
                if any(word in sentence for word in query.split()):
                    results.append(sentence)
                    break  # Stop searching for this document after finding a relevant sentence
    return results

# Streamlit app
def main():
    st.title("Multilingual Text Search Engine")

    # Upload dataset
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or TXT)", type=["csv", "txt"])

    if uploaded_file is not None:
        # Load dataset
        if uploaded_file.type == "text/plain":
            df = pd.read_csv(uploaded_file, sep="\t", header=None, names=["text"])
        else:
            df = pd.read_csv(uploaded_file)

        # Display uploaded data
        st.header("Uploaded Data")
        st.write(df.head())

        # Language selection
        lang = st.sidebar.selectbox("Select Language", ["Arabic", "English"])

        # Check language compatibility
        if lang.lower() == 'arabic':
            detected_lang = df['text'].apply(detect_language).mode().iloc[0]
            if detected_lang != 'ar':
                st.error("The detected language in the uploaded data is not Arabic. Please upload Arabic text.")
                return
        elif lang.lower() == 'english':
            detected_lang = df['text'].apply(detect_language).mode().iloc[0]
            if detected_lang != 'en':
                st.error("The detected language in the uploaded data is not English. Please upload English text.")
                return

        # Split text into sentences
        sentences = [sent_tokenize(text) for text in df['text']]

        # Data preprocessing
        st.sidebar.header("Data Preprocessing")
        if st.sidebar.checkbox("Clean Text"):
            df['text'] = df['text'].apply(clean_text)
            st.subheader("After Cleaning Text:")
            st.write(df.head())
        if st.sidebar.checkbox("Remove Stopwords"):
            df['text'] = df.apply(lambda row: remove_stopwords(row['text'], lang), axis=1)
            st.subheader("After Removing Stopwords:")
            st.write(df.head())

        # TF-IDF Vectorization
        X, vectorizer = tfidf_vectorizer(df['text'])

        # Search
        st.header("Search")
        query = st.text_input("Enter your search query")
        if st.button("Search"):
            if query:
                results = search(query, vectorizer, X, sentences)
                st.write("Search Results:")
                if results:
                    for result in results:
                        st.write(result)
                else:
                    st.write("No matching sentences found.")

if __name__ == "__main__":
    main()
