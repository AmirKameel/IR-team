import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

# Indexing Methods
def create_term_document_matrix(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def create_inverted_index(corpus):
    inverted_index = {}
    for idx, doc in enumerate(corpus):
        tokens = word_tokenize(doc)
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append(idx)
    return inverted_index

def create_tfidf_matrix(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

# Retrieval Methods
def retrieve_cosine_similarity(query, index, vectorizer, corpus):
    query = clean_text(query)
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, index).flatten()
    sorted_indices = cosine_similarities.argsort()[::-1]
    results = []
    for idx in sorted_indices:
        if cosine_similarities[idx] > 0:
            results.append(corpus[idx])
    return results

def retrieve_using_inverted_index(query, index, corpus):
    query = clean_text(query)
    tokens = word_tokenize(query)
    relevant_docs = set()
    for token in tokens:
        if token in index:
            relevant_docs.update(index[token])
    results = [corpus[idx] for idx in relevant_docs]
    return results

# Language detection
def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = None
    return lang

def highlight_query_in_results(query, result):
    # Case insensitive search and highlight with word boundaries
    query_regex = re.compile(r'\b' + re.escape(query) + r'\b', re.IGNORECASE)
    highlighted_result = query_regex.sub(lambda x: f"<span style='background-color: yellow; font-weight: bold;'>{x.group()}</span>", result)
    return highlighted_result

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

        # Indexing method selection
        indexing_method = st.sidebar.selectbox("Select Indexing Method", ["Term Document Matrix", "Inverted Index", "Tf-idf Vectorization"])

        # Indexing
        if indexing_method == "Term Document Matrix":
            index, vectorizer = create_term_document_matrix(df['text'])
        elif indexing_method == "Inverted Index":
            index = create_inverted_index(df['text'])
        elif indexing_method == "Tf-idf Vectorization":
            index, vectorizer = create_tfidf_matrix(df['text'])
        
        # Search
        st.header("Search")
        query = st.text_input("Enter your search query")
        retrieval_method = st.sidebar.selectbox("Select Retrieval Method", ["Cosine Similarity", "Using Inverted Index"])

        num_results = st.slider("Select the number of related documents to return", min_value=1, max_value=10, value=5)

        if st.button("Search"):
            if query:
                if indexing_method == "Term Document Matrix" or indexing_method == "Tf-idf Vectorization":
                    results = retrieve_cosine_similarity(query, index, vectorizer, df['text'])
                elif indexing_method == "Inverted Index":
                    results = retrieve_using_inverted_index(query, index, df['text'])
                st.write("Search Results:")
                if results:
                    num_results = min(len(results), num_results)
                    for i in range(num_results):
                        highlighted_result = highlight_query_in_results(query, results[i])
                        st.write(f"- {highlighted_result}", unsafe_allow_html=True)  # Allow HTML rendering
                else:
                    st.write("No matching sentences found.")

if __name__ == "__main__":
    main()
