import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import ISRIStemmer
from langdetect import detect
from collections import Counter
import math
import nltk
import PyPDF2
import io

import nltk
nltk.download()

# Arabic preprocessing functions
def clean_text(text, lang):
    if lang.lower() == 'arabic':
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Remove non-Arabic characters
        text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u0674\u06D6-\u06ED]', '', text)  # Remove Arabic diacritics
    elif lang.lower() == 'english':
        text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text

def remove_stopwords(text, lang):
    if lang.lower() == 'arabic':
        stop_words = set(stopwords.words('arabic'))
    elif lang.lower() == 'english':
        stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def stem_text(text, lang):
    if lang.lower() == 'arabic':
        stemmer = ISRIStemmer()
    elif lang.lower() == 'english':
        stemmer = nltk.PorterStemmer()
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

# Cosine similarity calculation
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = text.split()
    return Counter(words)

# Language detection
def detect_language(text):
    try:
        lang = detect(text)
        if lang == 'ar':
            return 'Arabic'
        elif lang == 'en':
            return 'English'
        else:
            return 'Unknown'
    except:
        return 'Unknown'

# Updated PDF reading function
def read_pdf(file, chunk_size=1000):
    pdf_reader = PyPDF2.PdfReader(file)
    chunks = []
    current_chunk = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        words = text.split()
        for word in words:
            current_chunk += word + " "
            if len(current_chunk.split()) >= chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Updated retrieve functions to work with chunks
def retrieve_cosine_similarity(query, index, corpus, lang):
    query = clean_text(query, lang)
    query_vec = text_to_vector(query)
    cosine_similarities = [(get_cosine(query_vec, text_to_vector(doc)), i) for i, doc in enumerate(corpus)]
    cosine_similarities.sort(reverse=True)
    results = [(cosine_score, corpus[idx], idx) for cosine_score, idx in cosine_similarities]
    return results

def retrieve_using_inverted_index(query, index, corpus, lang):
    query = clean_text(query, lang)
    tokens = word_tokenize(query)
    relevant_docs = set()
    for token in tokens:
        if token in index:
            relevant_docs.update(index[token])
    results = [(1.0, corpus[idx], idx) for idx in relevant_docs]  # Using 1.0 as a placeholder score
    return results

def highlight_query_in_results(query, result):
    try:
        # Case insensitive search and highlight with word boundaries
        query_regex = re.compile(r'\b' + re.escape(query) + r'\b', re.IGNORECASE)
        highlighted_result = query_regex.sub(lambda x: f"<span style='background-color: yellow; font-weight: bold;'>{x.group()}</span>", result)
        return highlighted_result
    except Exception as e:
        return result 

# Streamlit app
def main():
    st.title("MVP Aerosync")

    # Upload dataset
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file is not None:
        # Load dataset
        if uploaded_file.type == "text/plain":
            text_content = uploaded_file.getvalue().decode("utf-8")
            df = pd.DataFrame({"text": [text_content]})
        elif uploaded_file.type == "application/pdf":
            text_chunks = read_pdf(io.BytesIO(uploaded_file.getvalue()))
            df = pd.DataFrame({"text": text_chunks})

        # Display uploaded data
        st.header("Uploaded Data")
        st.write(f"Number of chunks: {len(df)}")
        st.write(df.head())

        # Language detection
        detected_lang = detect_language(df['text'].iloc[0])
        lang_options = ["Arabic", "English"]
        lang_index = lang_options.index(detected_lang) if detected_lang in lang_options else 0
        lang = st.sidebar.selectbox("Select Language", lang_options, index=lang_index)

        # Data preprocessing
        st.sidebar.header("Data Preprocessing")
        if st.sidebar.checkbox("Clean Text"):
            df['text'] = df['text'].apply(lambda x: clean_text(x, lang))
            st.subheader("After Cleaning Text:")
            st.write(df.head())
        if st.sidebar.checkbox("Remove Stopwords"):
            df['text'] = df['text'].apply(lambda x: remove_stopwords(x, lang))
            st.subheader("After Removing Stopwords:")
            st.write(df.head())

        # Check if there are any non-empty documents after preprocessing
        if df['text'].str.strip().astype(bool).any():
            # Indexing method selection
            indexing_method = st.sidebar.selectbox("Select Indexing Method", ["Tf-idf Vectorization", "Inverted Index", "Term Document Matrix"])

            # Indexing
            if indexing_method == "Tf-idf Vectorization":
                index, vectorizer = create_tfidf_matrix(df['text'])
            elif indexing_method == "Inverted Index":
                index = create_inverted_index(df['text'])
            elif indexing_method == "Term Document Matrix":
                index, vectorizer = create_term_document_matrix(df['text'])
            
            # Search
            st.header("Search")
            query = st.text_input("Enter your search query")
            retrieval_method = st.sidebar.selectbox("Select Retrieval Method", ["Cosine Similarity", "Using Inverted Index"])

            num_results = st.slider("Select the number of related chunks to return", min_value=1, max_value=10, value=5)

            if st.button("Search"):
                if query:
                    if indexing_method == "Term Document Matrix" or indexing_method == "Tf-idf Vectorization":
                        results = retrieve_cosine_similarity(query, index, df['text'], lang)
                    elif indexing_method == "Inverted Index":
                        results = retrieve_using_inverted_index(query, index, df['text'], lang)
                        
                    st.write("Search Results:")

                    if results:
                        num_results = min(len(results), num_results)
                        for i in range(num_results):        
                            score, text, chunk_id = results[i]

                            highlighted_result = highlight_query_in_results(query, text)
                            st.write(f"- Chunk ID: {chunk_id}, Score: {score:.4f}")
                            st.write(highlighted_result, unsafe_allow_html=True)
                            st.write("---")
                    else:
                        st.write("No matching chunks found.")
        else:
            st.error("All documents are empty after preprocessing. Please adjust preprocessing steps or upload a different dataset.")

if __name__ == "__main__":
    main()
