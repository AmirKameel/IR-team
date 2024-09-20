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

# Function to read and chunk PDFs
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

# Function to clean text (remove special characters)
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# Function to calculate cosine similarity
def calculate_similarity(chunk1, chunk2):
    vectorizer = TfidfVectorizer().fit_transform([chunk1, chunk2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return cosine_sim

# Streamlit app
def main():
    st.title("MVP AeroSync")

    # Initialize session state for buttons
    if 'similarity' not in st.session_state:
        st.session_state.similarity = None
    if 'audit_result' not in st.session_state:
        st.session_state.audit_result = None

    # Step 1: Upload PDFs
    st.sidebar.header("Upload PDFs")
    uploaded_file1 = st.sidebar.file_uploader("Upload first PDF", type="pdf")
    uploaded_file2 = st.sidebar.file_uploader("Upload second PDF", type="pdf")

    if uploaded_file1 and uploaded_file2:
        # Read and chunk PDFs
        text_chunks1 = read_pdf(io.BytesIO(uploaded_file1.getvalue()))
        text_chunks2 = read_pdf(io.BytesIO(uploaded_file2.getvalue()))

        df1 = pd.DataFrame({"text": text_chunks1})
        df2 = pd.DataFrame({"text": text_chunks2})

        st.header("Uploaded Data")
        st.write("First PDF Chunks:")
        st.write(df1.head())
        st.write("Second PDF Chunks:")
        st.write(df2.head())

        # Step 2: Select chunks for comparison
        st.sidebar.subheader("Select Chunks to Compare")
        chunk1_idx = st.sidebar.selectbox("Select chunk from first PDF", range(len(text_chunks1)))
        chunk2_idx = st.sidebar.selectbox("Select chunk from second PDF", range(len(text_chunks2)))

        chunk1 = df1['text'].iloc[chunk1_idx]
        chunk2 = df2['text'].iloc[chunk2_idx]

        st.subheader("Selected Chunks for Comparison")
        st.write("**First PDF Chunk:**")
        st.write(chunk1)
        st.write("**Second PDF Chunk:**")
        st.write(chunk2)

        # Step 3: Calculate similarity
        if st.button("Compute Similarity"):
            cleaned_chunk1 = clean_text(chunk1)
            cleaned_chunk2 = clean_text(chunk2)
            similarity = calculate_similarity(cleaned_chunk1, cleaned_chunk2)
            st.session_state.similarity = similarity  # Save to session state

        # Display similarity if calculated
        if st.session_state.similarity is not None:
            st.write(f"Cosine Similarity: {st.session_state.similarity:.4f}")

        # Step 4: Perform audit using OpenAI GPT
        if st.button("Perform Audit"):
            audit_result = perform_audit(chunk1, chunk2)
            st.session_state.audit_result = audit_result  # Save to session state

        # Display audit result if calculated
        if st.session_state.audit_result is not None:
            st.subheader("Audit Result")
            st.write(st.session_state.audit_result)

if __name__ == "__main__":
    main()
