import os
import json
import logging
import streamlit as st
from scripts.ResumeProcessor import ResumeProcessor
from scripts.JobDescriptionProcessor import JobDescriptionProcessor
from scripts.utils.logger import init_logging_config
from scripts.utils.ReadFiles import get_filenames_from_dir
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

init_logging_config()

PROCESSED_RESUMES_PATH = "Data/Processed/Resumes"
PROCESSED_JOB_DESCRIPTIONS_PATH = "Data/Processed/JobDescription"
UPLOADED_RESUMES_PATH = "Data/Resumes"  

def read_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def remove_old_files(files_path):
    for filename in os.listdir(files_path):
        try:
            file_path = os.path.join(files_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.error(f"Error deleting {file_path}:\n{e}")

    logging.info("Deleted old files from " + files_path)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalnum()]
    return " ".join(tokens)

def get_similarity_score(resume_string, job_description_string):
    resume_string = preprocess_text(resume_string)
    job_description_string = preprocess_text(job_description_string)
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([resume_string, job_description_string])
    
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return similarity_score

def check_eligibility(resume_string, job_description_files, threshold=0.1):
    eligible_jobs = []
    for job_file in job_description_files:
        job_description = read_json(os.path.join(PROCESSED_JOB_DESCRIPTIONS_PATH,job_file))
        job_description_text = ' '.join(job_description["extracted_keywords"])
        similarity_score = get_similarity_score(resume_string, job_description_text)
        if similarity_score > threshold:
            eligible_jobs.append(job_file)
    return eligible_jobs

def main():
    st.set_page_config(
        page_title="iamneo - AI Gatekeeper",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.image("https://images.crunchbase.com/image/upload/c_lpad,f_auto,q_auto:eco,dpr_1/ngtdnfed3au9ifom5w5b", width=180)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(to right bottom, #eeeee4, #c6d1c4, #9cb5ac, #74999a, #537c8b, #4c6f82, #476378, #43566d, #485769, #4d5865, #525961, #575a5d);
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    new_title = '<p style="font-family:KaTeX_Caligraphic; color:#d2dacd; font-size: 45px; text-align: center; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);"><strong>Your AI Gatekeeper</strong></p>'
    st.markdown(new_title, unsafe_allow_html=True)

    new_header = '<p style="font-family:sans-serif; color:Black; font-size: 20px; text-align: center; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);"><i>An ATS to help your resume pass the <strong>Screening Stage.</strong></i></p>'
    st.markdown(new_header, unsafe_allow_html=True)

    uploaded_resume = st.file_uploader(" ",type=["pdf", "txt"])

    eligible_jobs = [] 
    success = False
    if uploaded_resume is not None:
        with st.spinner("Processing your resume..."):
            if not os.path.exists(UPLOADED_RESUMES_PATH):
                os.makedirs(UPLOADED_RESUMES_PATH)
            remove_old_files(UPLOADED_RESUMES_PATH)
            user_resume_path = os.path.join(UPLOADED_RESUMES_PATH, "user_resume.pdf")
            with open(user_resume_path, "wb") as f:
                f.write(uploaded_resume.read())

            remove_old_files(PROCESSED_RESUMES_PATH)
            processor = ResumeProcessor("user_resume.pdf")
            success = processor.process()

            if success:
                user_resume_file = os.path.join(PROCESSED_RESUMES_PATH, "user_resume.pdf-processed.json")
                if os.path.exists(user_resume_file):
                    user_resume = read_json(user_resume_file)
                    user_resume_keywords = ' '.join(user_resume["extracted_keywords"])
                    job_description_files = get_filenames_from_dir(PROCESSED_JOB_DESCRIPTIONS_PATH)
                    eligible_jobs = check_eligibility(user_resume_keywords, job_description_files)
                else:
                    st.error("Error: User's processed resume file not found.")
            else:
                st.error("Error: Failed to process user's resume.")
    else:
        st.warning('Please upload your resume in .pdf or .txt format to check eligibility for the below Roles.', icon="🤖")
        job_description_files = get_filenames_from_dir(PROCESSED_JOB_DESCRIPTIONS_PATH)
        all_roles = '<p style="font-family:sans-serif; color:Black; font-size: 20px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);"><i>The Roles for Which We Assess Your <strong>Profile !!</strong></i></p>'
        st.markdown(all_roles, unsafe_allow_html=True)
        for job_file in job_description_files:
         job_name = os.path.basename(job_file).split(".")[0]
         st.markdown(f'<span style="color: black; font-size: 18px;"><em><strong>~  {job_name}</em></strong></span>', unsafe_allow_html=True)

    if success and eligible_jobs:
        st.markdown('<p style="font-family:sans-serif; color:Black; font-size: 20px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);"><i>Eligible Jobs for you Based on your <strong>Resume !!</strong></i></p>', unsafe_allow_html=True)
        for job_file in eligible_jobs:
            job_name = os.path.basename(job_file).split(".")[0]
            st.markdown(f'<span style="color: black; font-size: 18px;"><em><strong>~  {job_name}</em></strong></span>', unsafe_allow_html=True)
    elif success:
        st.markdown('<p style="font-family:sans-serif; color:Black; font-size: 18px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);"><em>You\'re not Eligible for any of the <strong>Jobs !!</strong><br>Just Keep building your Resume, Don\'t Lose Hope ❤️</em></p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
