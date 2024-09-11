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
import openai
import toml



# Set your OpenAI API key here



def perform_audit(iosa_checklist, input_text):
    model_id = 'gpt-4o'  
    # Load the secrets from the toml file
    secrets = toml.load('secrets.toml')

    # Create the OpenAI client using the API key from secrets.toml
    client = openai.OpenAI(api_key=secrets['openai']['api_key'])

    # OpenAI API request
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                'role': 'system',
                'content': (
                    """
    Context	Your role is pivotal as you conduct audits to ensure strict compliance with ISARPs. Your meticulous evaluation of legal documents against ISARPs is crucial. We entrust you with the responsibility of upholding legal standards in the aviation industry. During an audit, an operator is assessed against the ISARPs contained in this manual. To determine conformity with any standard or recommended practice, an auditor will gather evidence to assess the degree to which specifications are documented and implemented by the operator. In making such an assessment, the following information is applicable.	You're an aviation professional with a robust 20-year background in both the business and commercial sectors of the industry. Your expertise extends to a deep-rooted understanding of aviation regulations the world over, a strong grasp of safety protocols, and a keen perception of the regulatory differences that come into play internationally.
    Your experience is underpinned by a solid educational foundation and specialized professional training. This has equipped you with a thorough and detailed insight into the technical and regulatory dimensions of aviation. Your assessments are carried out with attention to detail and a disciplined use of language that reflects a conscientious approach to legal responsibilities.
    In your role, you conduct audits of airlines to ensure they align with regulatory mandates, industry benchmarks, and established best practices. You approach this task with a critical eye, paying close attention to the language used and its implications. It's your job to make sure that terminology is employed accurately in compliance with legal stipulations.
    From a technical standpoint, your focus is on precise compliance with standards, interpreting every word of regulatory requirements and standards literally and ensuring these are fully reflected within the airline's legal documentation.
    In the realm of aviation, you are recognized as a font of knowledge, possessing a breadth of experience that stretches across various departments within an aviation organization.
    Your task involves meticulously evaluating the airline's legal documents against these benchmarks, verifying that the responses provided meet the stipulated regulations or standards. You then present a detailed assessment, thoroughly outlining both strong points and areas needing improvement, and offering actionable advice for enhancements.
    Your approach to evaluating strengths and weaknesses is methodical, employing legal terminology with a level of precision and detail akin to that of a seasoned legal expert.
    Furthermore, if requested, you are adept at supplementing statements in such a way that they comprehensively address and fulfill the relevant regulatory requirements or standards, ensuring complete compliance and thoroughness in documentation.
    """
      )
            },
            {
                'role': 'user',
                'content': (
                    f"""
    OBJECTIVES:
    you are given a doc and a input text do the followings:
    The provided text is to be evaluated on a compliance scale against the requirements of the regulatory text or international standard, ranging from 0 to 10. A score of 0 indicates the text is entirely non-compliant or irrelevant to the set requirements, while a score of 10 denotes full compliance with the specified criteria.
    The text's relevance and adherence to the given standards must be analyzed, and an appropriate score within this range should be assigned based on the assessment.
    Provide a thorough justification for the assigned score. Elaborate on the specific factors and criteria that influenced your decision, detailing how the text meets or fails to meet the established requirements, which will support the numerical compliance rating you have provided
    Should your assessment yield a compliance score greater than 3, you should provide supplemental text to the original content, drawing from industry best practices and benchmarks, as well as referencing pertinent regulatory materials or standards. The supplementary text should be crafted in a human writing style, incorporating human factors principles to ensure it is clear, readable, and easily understood by crew members. It's important to note that aviation regulations emphasize ease of language and precision in communication.
    In the case where the provided text is deemed completely irrelevant, you are to utilize your expertise, industry benchmarks, best practices, and relevant regulatory references or standards to formulate a detailed exposition of processes, procedures, organizational structure, duty management, or any other facet within the aviation industry. The goal is to revise the text to achieve full compliance with the applicable legal requirements or standards.

    ISARPs: 
    {iosa_checklist}
    INPUT_TEXT: 
    {input_text}

    Your output must include the following sections:
    ASSESSMENT: A detailed evaluation of the documentation's alignment with the ISARPs. It should employ technical language and aviation terminology where appropriate.
    RECOMMENDATIONS: Specific, actionable suggestions aimed at improving compliance with ISARP standards. Maintain a formal and professional tone.
    OVERALL_COMPLIANCE_SCORE: A numerical rating (0 to 10) reflecting the documentation's overall compliance with the ISARPs.
    OVERALL_COMPLIANCE_TAG: A scoring tag indicating the overall compliance level with ISARPs.
    """
    )
            }
        ],
        max_tokens=4000
    )
    
    return response.choices[0].message.content




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

def main():
    st.title("MVP Aerosync")

    # Initialize session state
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []

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
 

        if df['text'].str.strip().astype(bool).any():
            indexing_method = st.sidebar.selectbox("Select Indexing Method", ["Tf-idf Vectorization", "Inverted Index", "Term Document Matrix"])

            if indexing_method == "Tf-idf Vectorization":
                index, vectorizer = create_tfidf_matrix(df['text'])
            elif indexing_method == "Inverted Index":
                index = create_inverted_index(df['text'])
            elif indexing_method == "Term Document Matrix":
                index, vectorizer = create_term_document_matrix(df['text'])
            
            st.header("Search")
            query = st.text_input("Enter your search query")
            retrieval_method = st.sidebar.selectbox("Select Retrieval Method", ["Cosine Similarity", "Using Inverted Index"])

            num_results = st.slider("Select the number of related chunks to return", min_value=1, max_value=10, value=5)

            if st.button("Search"):
                if query:
                    if indexing_method in ["Term Document Matrix", "Tf-idf Vectorization"]:
                        results = retrieve_cosine_similarity(query, index, df['text'], lang)
                    elif indexing_method == "Inverted Index":
                        results = retrieve_using_inverted_index(query, index, df['text'], lang)
                    
                    st.session_state.search_results = results
                    st.session_state.search_performed = True

            # Display search results and audit buttons
            if st.session_state.search_performed:
                st.write("Search Results:")
                if st.session_state.search_results:
                    num_results = min(len(st.session_state.search_results), num_results)
                    for i in range(num_results):        
                        score, text, chunk_id = st.session_state.search_results[i]

                        highlighted_result = highlight_query_in_results(query, text)
                        st.write(f"- Chunk ID: {chunk_id}, Score: {score:.4f}")
                        st.write(highlighted_result, unsafe_allow_html=True)
                        
                        # Add unique key to each button
                        if st.button(f'Audit Chunk {chunk_id}', key=f"audit_button_{chunk_id}"):
                            audit_result = perform_audit(highlighted_result, query)
                            st.subheader(f"Audit Results for Chunk {chunk_id}")
                            st.write(audit_result)
                        st.write("---")
                else:
                    st.write("No matching chunks found.")
        else:
            st.error("All documents are empty after preprocessing. Please adjust preprocessing steps or upload a different dataset.")

if __name__ == "__main__":
    main()
